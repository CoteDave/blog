#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║           dash_to_html.py — Dash App → Portable HTML                ║
║                                                                      ║
║  Convertit n'importe quelle app Dash en un fichier HTML autonome,   ║
║  pleinement dynamique et portable (zéro dépendance serveur).         ║
║                                                                      ║
║  Stratégies de portabilité :                                         ║
║  1. Tous les assets CSS/JS/fonts/images sont inlinés                 ║
║  2. Les callbacks sont pré-calculés pour toutes les combinaisons     ║
║     d'inputs possibles → table de lookup JS injectée                 ║
║  3. Les figures Plotly sont sérialisées en JSON et rechargées        ║
║     via Plotly.js en mode offline                                    ║
║  4. Fallback Pyodide disponible pour les callbacks complexes         ║
║                                                                      ║
║  Usage :                                                             ║
║    python dash_to_html.py app.py                                     ║
║    python dash_to_html.py app.py -o rapport.html --port 8052        ║
║    python dash_to_html.py app.py --pyodide --max-combos 500         ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import argparse
import base64
import importlib.util
import itertools
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dash2html")


# ── Helpers ───────────────────────────────────────────────────────────────────

def b64_encode(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def guess_mime(url: str, content: bytes) -> str:
    ext = Path(urlparse(url).path).suffix.lower()
    mime_map = {
        ".woff": "font/woff", ".woff2": "font/woff2",
        ".ttf": "font/ttf", ".eot": "application/vnd.ms-fontobject",
        ".svg": "image/svg+xml", ".png": "image/png",
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".ico": "image/x-icon",
        ".css": "text/css", ".js": "application/javascript",
        ".json": "application/json",
    }
    return mime_map.get(ext, mimetypes.guess_type(url)[0] or "application/octet-stream")


def fetch(url: str, session: requests.Session, timeout: int = 30) -> bytes | None:
    try:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        return r.content
    except Exception as e:
        log.warning(f"  ⚠  Impossible de récupérer {url} : {e}")
        return None


# ── App Loader ─────────────────────────────────────────────────────────────────

def load_dash_app(app_path: str):
    """
    Importe le fichier Python de l'app Dash.
    Retourne le module et l'objet `app` (cherche les noms courants).
    """
    app_path = os.path.abspath(app_path)
    app_dir  = os.path.dirname(app_path)

    # Ajoute le dossier de l'app au sys.path (pour les imports relatifs)
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)

    spec = importlib.util.spec_from_file_location("_dash_app_module_", app_path)
    module = importlib.util.module_from_spec(spec)

    # Empêche l'app de démarrer automatiquement si elle a un if __name__ == "__main__"
    module.__name__ = "_dash_app_module_"

    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass  # Certaines apps appellent sys.exit() — on ignore

    # Cherche l'objet Dash (noms courants)
    for candidate in ("app", "application", "server", "dash_app"):
        obj = getattr(module, candidate, None)
        if obj is not None:
            type_name = type(obj).__name__
            if "Dash" in type_name or "Flask" in type_name:
                log.info(f"  ✓  App Dash trouvée → variable '{candidate}' ({type_name})")
                return module, obj

    raise ValueError(
        "Impossible de trouver l'objet Dash dans le fichier. "
        "Assurez-vous qu'il s'appelle `app`, `application` ou `server`."
    )


# ── Server ─────────────────────────────────────────────────────────────────────

def start_server(app, port: int) -> threading.Thread:
    """Lance le serveur Dash dans un thread daemon."""
    def _run():
        try:
            app.run(port=port, debug=False, use_reloader=False)
        except Exception:
            # Certaines versions de Dash utilisent run_server
            try:
                app.run_server(port=port, debug=False, use_reloader=False)
            except Exception as e:
                log.error(f"Erreur serveur : {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def wait_for_server(port: int, timeout: int = 30) -> bool:
    """Attend que le serveur réponde (polling)."""
    url = f"http://127.0.0.1:{port}/"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(0.4)
    return False


# ── Asset Inliner ──────────────────────────────────────────────────────────────

class AssetInliner:
    """
    Télécharge et inline tous les assets externes :
    CSS, JavaScript, polices, images.
    Remplace les URLs par des data-URIs ou du texte inline.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session  = requests.Session()
        self.cache: dict[str, bytes] = {}

    def _get(self, url: str) -> bytes | None:
        full_url = urljoin(self.base_url, url) if not url.startswith("http") else url
        if full_url in self.cache:
            return self.cache[full_url]
        data = fetch(full_url, self.session)
        if data:
            self.cache[full_url] = data
        return data

    def _inline_css_assets(self, css_text: str, css_base_url: str) -> str:
        """Inline les url() dans un fichier CSS (fonts, images)."""
        def replace_url(m):
            raw = m.group(1).strip("'\"")
            if raw.startswith("data:") or raw.startswith("#"):
                return m.group(0)
            abs_url = urljoin(css_base_url, raw)
            data = fetch(abs_url, self.session)
            if data:
                mime = guess_mime(abs_url, data)
                return f"url('{b64_encode(data, mime)}')"
            return m.group(0)
        return re.sub(r"url\(([^)]+)\)", replace_url, css_text)

    def inline_all(self, soup: BeautifulSoup) -> BeautifulSoup:
        log.info("📦  Inline des assets CSS …")
        self._inline_stylesheets(soup)
        log.info("📦  Inline des assets JavaScript …")
        self._inline_scripts(soup)
        log.info("📦  Inline des images …")
        self._inline_images(soup)
        return soup

    def _inline_stylesheets(self, soup: BeautifulSoup):
        for link in soup.find_all("link", rel=lambda v: v and "stylesheet" in v):
            href = link.get("href", "")
            if not href or href.startswith("data:"):
                continue
            full_url = urljoin(self.base_url, href)
            data = self._get(full_url)
            if data:
                css_text = data.decode("utf-8", errors="replace")
                css_text = self._inline_css_assets(css_text, full_url)
                style = soup.new_tag("style")
                style.string = css_text
                link.replace_with(style)
                log.debug(f"    CSS inliné : {href}")

    def _inline_scripts(self, soup: BeautifulSoup):
        for script in soup.find_all("script", src=True):
            src = script.get("src", "")
            if not src or src.startswith("data:"):
                continue
            full_url = urljoin(self.base_url, src)
            data = self._get(full_url)
            if data:
                new_s = soup.new_tag("script")
                # Préserve les attributs (type, defer, etc.) sauf src
                for k, v in script.attrs.items():
                    if k != "src":
                        new_s[k] = v
                new_s.string = data.decode("utf-8", errors="replace")
                script.replace_with(new_s)
                log.debug(f"    JS inliné : {src}")

    def _inline_images(self, soup: BeautifulSoup):
        for img in soup.find_all("img", src=True):
            src = img.get("src", "")
            if src.startswith("data:") or not src:
                continue
            full_url = urljoin(self.base_url, src)
            data = self._get(full_url)
            if data:
                mime = guess_mime(full_url, data)
                img["src"] = b64_encode(data, mime)


# ── Callback Pre-computer ─────────────────────────────────────────────────────

class CallbackPrecomputer:
    """
    Interagit avec l'API interne Dash pour :
    1. Lister tous les callbacks (inputs, outputs, states)
    2. Énumérer toutes les combinaisons d'inputs possibles
    3. Appeler le serveur pour chaque combinaison
    4. Construire une table de lookup JSON
    """

    def __init__(self, base_url: str, max_combos: int = 300):
        self.base_url   = base_url
        self.max_combos = max_combos
        self.session    = requests.Session()

    def _get_layout(self) -> dict:
        r = self.session.get(urljoin(self.base_url, "/_dash-layout"), timeout=10)
        return r.json()

    def _get_dependencies(self) -> list[dict]:
        r = self.session.get(urljoin(self.base_url, "/_dash-dependencies"), timeout=10)
        return r.json()

    def _extract_component_options(self, layout: dict, comp_id: str, prop: str) -> list[Any]:
        """Extrait récursivement les options/valeurs possibles d'un composant."""
        results = []
        self._walk_layout(layout, comp_id, prop, results)
        return results

    def _walk_layout(self, node: Any, target_id: str, prop: str, acc: list):
        if not isinstance(node, dict):
            return
        props = node.get("props", {})
        node_id = props.get("id")

        if node_id == target_id or (
            isinstance(node_id, dict) and node_id == target_id
        ):
            if prop == "value" and "options" in props:
                options = props["options"]
                if isinstance(options, list):
                    acc.extend(
                        o.get("value", o) if isinstance(o, dict) else o
                        for o in options
                    )
            elif prop in props:
                val = props[prop]
                if isinstance(val, list):
                    acc.extend(val)
                else:
                    acc.append(val)
            # Pour les sliders : min/max/step
            if prop == "value" and "min" in props and "max" in props:
                mn  = props["min"]
                mx  = props["max"]
                stp = props.get("step", 1)
                if stp and (mx - mn) / stp <= 50:
                    v = mn
                    while v <= mx:
                        if v not in acc:
                            acc.append(round(v, 6))
                        v += stp

        for child in props.get("children", []):
            self._walk_layout(child, target_id, prop, acc)

    def _call_callback(self, output_id: str, output_prop: str,
                       inputs: list[dict], states: list[dict],
                       input_values: dict[str, Any]) -> Any:
        """Envoie une requête de callback au serveur Dash."""
        payload = {
            "output":        f"{output_id}.{output_prop}",
            "outputs":       {"id": output_id, "property": output_prop},
            "changedPropIds": [],
            "inputs":  [
                {
                    "id":       inp["id"] if isinstance(inp["id"], str) else json.dumps(inp["id"]),
                    "property": inp["property"],
                    "value":    input_values.get(f"{inp['id']}.{inp['property']}"),
                }
                for inp in inputs
            ],
            "state": [
                {
                    "id":       st["id"] if isinstance(st["id"], str) else json.dumps(st["id"]),
                    "property": st["property"],
                    "value":    input_values.get(f"{st['id']}.{st['property']}"),
                }
                for st in states
            ],
        }
        try:
            r = self.session.post(
                urljoin(self.base_url, "/_dash-update-component"),
                json=payload,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            log.debug(f"    Callback échoué : {e}")
        return None

    def precompute(self) -> dict:
        """
        Retourne un dict de la forme :
        {
          "<output_id>.<output_prop>": {
            "<input_key_hash>": <response_value>,
            ...
          },
          ...
        }
        """
        log.info("🔄  Récupération du layout et des dépendances …")
        try:
            layout = self._get_layout()
            deps   = self._get_dependencies()
        except Exception as e:
            log.warning(f"  ⚠  Impossible de récupérer les dépendances : {e}")
            return {}

        lookup: dict[str, dict] = {}
        total_callbacks = len(deps)
        log.info(f"  → {total_callbacks} callback(s) détecté(s)")

        for cb_idx, cb in enumerate(deps):
            outputs = cb.get("output", "")
            inputs  = cb.get("inputs",  [])
            states  = cb.get("state",   [])

            # Parse les outputs (peut être multi-output)
            output_entries = self._parse_outputs(outputs)

            # Collecte les valeurs possibles pour chaque input
            input_options: list[list[tuple[str, Any]]] = []
            for inp in inputs:
                inp_id  = inp["id"] if isinstance(inp["id"], str) else json.dumps(inp["id"])
                inp_key = f"{inp_id}.{inp['property']}"
                options = self._extract_component_options(layout, inp["id"], inp["property"])
                if not options:
                    # Valeur par défaut si aucune option trouvée
                    options = [None]
                input_options.append([(inp_key, v) for v in options])

            # Calcule le nombre de combinaisons
            total_combos = 1
            for opts in input_options:
                total_combos *= len(opts)

            if total_combos > self.max_combos:
                log.warning(
                    f"  ⚠  Callback {cb_idx+1}/{total_callbacks} : {total_combos} combinaisons "
                    f"> max ({self.max_combos}) — ignoré (utilisez --max-combos pour augmenter)"
                )
                continue

            log.info(
                f"  ⟳  Callback {cb_idx+1}/{total_callbacks} : "
                f"{total_combos} combinaison(s) × {len(output_entries)} output(s)"
            )

            for combo in itertools.product(*input_options):
                input_values = dict(combo)
                combo_key    = json.dumps(input_values, sort_keys=True)

                result = self._call_callback(
                    output_entries[0][0], output_entries[0][1],
                    inputs, states, input_values
                )
                if result is None:
                    continue

                response_data = result.get("response", {})
                for out_id, out_prop in output_entries:
                    lk_key = f"{out_id}.{out_prop}"
                    if lk_key not in lookup:
                        lookup[lk_key] = {}
                    # Cherche la valeur dans la réponse
                    val = (
                        response_data
                        .get(out_id, {})
                        .get("props", {})
                        .get(out_prop)
                    )
                    if val is None and "response" in result:
                        # Format multi-output
                        for key, data in response_data.items():
                            if out_prop in data.get("props", {}):
                                val = data["props"][out_prop]
                                break
                    lookup[lk_key][combo_key] = val

        n_entries = sum(len(v) for v in lookup.values())
        log.info(f"  ✓  Table de lookup : {len(lookup)} output(s), {n_entries} entrée(s)")
        return lookup

    @staticmethod
    def _parse_outputs(outputs_str: str) -> list[tuple[str, str]]:
        """Parse '..' ou liste de 'id.prop'."""
        entries = []
        if not outputs_str:
            return entries
        # Multi-output : ..id1.prop1..id2.prop2..
        pattern = re.findall(r"\.\.([^.]+)\.([^.]+)\.\.", ".." + outputs_str + "..")
        if pattern:
            entries = [(i, p) for i, p in pattern]
        else:
            # Single output : id.prop
            parts = outputs_str.rsplit(".", 1)
            if len(parts) == 2:
                entries = [(parts[0], parts[1])]
        return entries


# ── JS Callback Injector ───────────────────────────────────────────────────────

CALLBACK_JS_TEMPLATE = """
/* ── dash2html Callback Engine ── */
(function() {
  'use strict';

  const LOOKUP = {lookup_json};

  /* Sérialisation des valeurs d'input → clé de lookup */
  function makeKey(inputValues) {{
    return JSON.stringify(inputValues, Object.keys(inputValues).sort());
  }}

  /* Applique une valeur à un composant React/Dash */
  function applyOutput(outputKey, value) {{
    if (value === undefined || value === null) return;
    const [compId, prop] = outputKey.split(/\\.(?=[^.]+$)/);

    // Figures Plotly
    if (prop === 'figure' && value && value.data !== undefined) {{
      const el = document.getElementById(compId);
      if (el) {{
        const graphDiv = el.querySelector('.js-plotly-plot') || el;
        if (window.Plotly && graphDiv._fullLayout !== undefined) {{
          Plotly.react(graphDiv, value.data || [], value.layout || {{}}, value.config || {{}});
        }}
      }}
      return;
    }}

    // Texte / HTML
    if (prop === 'children' || prop === 'value') {{
      const el = document.getElementById(compId);
      if (!el) return;
      if (typeof value === 'string' || typeof value === 'number') {{
        el.textContent = value;
      }} else if (Array.isArray(value) || typeof value === 'object') {{
        try {{
          el.innerHTML = JSON.stringify(value);
        }} catch(e) {{}}
      }}
      return;
    }}

    // Options dropdown
    if (prop === 'options') {{
      const el = document.getElementById(compId);
      if (el && el.tagName === 'SELECT') {{
        el.innerHTML = '';
        (value || []).forEach(function(opt) {{
          const o = document.createElement('option');
          o.value = opt.value !== undefined ? opt.value : opt;
          o.textContent = opt.label !== undefined ? opt.label : opt;
          el.appendChild(o);
        }});
      }}
      return;
    }}

    // Style
    if (prop === 'style' && typeof value === 'object') {{
      const el = document.getElementById(compId);
      if (el) Object.assign(el.style, value);
      return;
    }}

    // className
    if (prop === 'className') {{
      const el = document.getElementById(compId);
      if (el) el.className = value;
      return;
    }}

    // Fallback : attribut DOM
    const el = document.getElementById(compId);
    if (el) {{
      try {{ el.setAttribute(prop, value); }} catch(e) {{}}
    }}
  }}

  /* Collecte les valeurs courantes de tous les composants connus */
  function collectInputValues(inputKeys) {{
    const values = {{}};
    inputKeys.forEach(function(key) {{
      const [compId, prop] = key.split(/\\.(?=[^.]+$)/);
      const el = document.getElementById(compId);
      if (!el) return;
      if (prop === 'value') {{
        values[key] = el.value !== undefined ? el.value : null;
      }} else if (prop === 'n_clicks') {{
        values[key] = parseInt(el.dataset.nClicks || '0', 10);
      }} else {{
        values[key] = el.getAttribute(prop);
      }}
    }});
    return values;
  }}

  /* Déclenche tous les callbacks concernés par un output donné */
  function triggerCallbacks(changedInputKey) {{
    Object.keys(LOOKUP).forEach(function(outputKey) {{
      const table = LOOKUP[outputKey];
      // Vérifie si cet output dépend de l'input modifié
      const sampleKey = Object.keys(table)[0];
      if (!sampleKey) return;
      const sampleInputs = JSON.parse(sampleKey);
      if (!(changedInputKey in sampleInputs)) return;

      // Collecte les valeurs courantes
      const inputValues = collectInputValues(Object.keys(sampleInputs));
      const lookupKey   = makeKey(inputValues);
      const result      = table[lookupKey];
      if (result !== undefined) {{
        applyOutput(outputKey, result);
      }}
    }});
  }}

  /* Attache les listeners après que Dash ait monté les composants */
  function attachListeners() {{
    /* Dropdowns / Selects natifs */
    document.querySelectorAll('select').forEach(function(el) {{
      el.addEventListener('change', function() {{
        triggerCallbacks(el.id + '.value');
      }});
    }});

    /* Input range (sliders natifs) */
    document.querySelectorAll('input[type="range"]').forEach(function(el) {{
      el.addEventListener('input', function() {{
        triggerCallbacks(el.id + '.value');
      }});
    }});

    /* Boutons */
    document.querySelectorAll('button').forEach(function(el) {{
      el.addEventListener('click', function() {{
        el.dataset.nClicks = (parseInt(el.dataset.nClicks || '0', 10) + 1).toString();
        triggerCallbacks(el.id + '.n_clicks');
      }});
    }});

    /* Observe les nouveaux éléments Dash (React montés après load) */
    const observer = new MutationObserver(function(mutations) {{
      let changed = false;
      mutations.forEach(function(m) {{
        if (m.addedNodes.length > 0) changed = true;
      }});
      if (changed) {{
        setTimeout(attachDashComponentListeners, 300);
      }}
    }});
    observer.observe(document.body, {{ childList: true, subtree: true }});
  }}

  /* Listeners pour les composants Dash React (Dropdowns dcc, Sliders, etc.) */
  function attachDashComponentListeners() {{
    /* dcc.Dropdown (Select2/React-Select) */
    document.querySelectorAll('.Select-input input, .dash-dropdown input').forEach(function(el) {{
      if (el._dash2html_attached) return;
      el._dash2html_attached = true;
      const container = el.closest('[id]');
      if (!container) return;
      el.addEventListener('change', function() {{
        triggerCallbacks(container.id + '.value');
      }});
    }});

    /* dcc.Slider (RC-Slider) */
    document.querySelectorAll('.rc-slider').forEach(function(el) {{
      if (el._dash2html_attached) return;
      el._dash2html_attached = true;
      const container = el.closest('[id]');
      if (!container) return;
      el.addEventListener('mouseup', function() {{
        setTimeout(function() {{ triggerCallbacks(container.id + '.value'); }}, 50);
      }});
    }});
  }}

  /* Initialisation après chargement complet */
  window.addEventListener('DOMContentLoaded', function() {{
    setTimeout(function() {{
      attachListeners();
      attachDashComponentListeners();
      /* Déclenche un rendu initial pour tous les outputs connus */
      Object.keys(LOOKUP).forEach(function(outputKey) {{
        const table = LOOKUP[outputKey];
        const keys  = Object.keys(table);
        if (keys.length > 0) {{
          /* Utilise la première combinaison comme état initial */
          applyOutput(outputKey, table[keys[0]]);
        }}
      }});
      console.log('[dash2html] Callback engine initialisé — ' +
        Object.keys(LOOKUP).length + ' output(s) actif(s).');
    }}, 800);
  }});
}());
"""


def inject_callback_engine(soup: BeautifulSoup, lookup: dict) -> BeautifulSoup:
    """Injecte le moteur de callbacks JS dans le HTML."""
    if not lookup:
        log.info("  ℹ  Aucun callback à injecter.")
        return soup

    lookup_json = json.dumps(lookup, ensure_ascii=False, separators=(",", ":"))
    js_code = CALLBACK_JS_TEMPLATE.replace("{lookup_json}", lookup_json)

    script = soup.new_tag("script")
    script.string = js_code
    body = soup.find("body")
    if body:
        body.append(script)
    else:
        soup.append(script)

    log.info(f"  ✓  Moteur de callbacks injecté ({len(lookup)} output(s))")
    return soup


# ── Offline Banner ─────────────────────────────────────────────────────────────

OFFLINE_BANNER_CSS = """
<style>
#_dash2html_banner {
  position: fixed; bottom: 12px; right: 14px; z-index: 99999;
  background: #1a2540; color: #cbd5e1; font: 11px/1.4 monospace;
  padding: 8px 12px; border-radius: 6px; border-left: 3px solid #ff6b35;
  box-shadow: 0 2px 12px rgba(0,0,0,0.4); opacity: 0.88;
  max-width: 260px;
}
#_dash2html_banner strong { color: #ff6b35; }
</style>
"""

OFFLINE_BANNER_HTML = """
<div id="_dash2html_banner">
  <strong>dash2html</strong> — mode hors-ligne<br>
  Callbacks pré-calculés · Assets inlinés<br>
  <span style="color:#22c55e">✓ Portable &amp; autonome</span>
</div>
"""


def inject_banner(soup: BeautifulSoup) -> BeautifulSoup:
    head = soup.find("head")
    if head:
        head.append(BeautifulSoup(OFFLINE_BANNER_CSS, "html.parser"))
    body = soup.find("body")
    if body:
        body.append(BeautifulSoup(OFFLINE_BANNER_HTML, "html.parser"))
    return soup


# ── Pyodide Fallback ───────────────────────────────────────────────────────────

PYODIDE_TEMPLATE = """
<script>
/* ── dash2html : Pyodide Fallback (callbacks complexes) ── */
(async function() {{
  const statusEl = document.createElement('div');
  statusEl.style.cssText = 'position:fixed;top:10px;right:10px;background:#1a2540;'
    + 'color:#fbbf24;padding:8px 12px;border-radius:6px;font:12px monospace;z-index:99999';
  statusEl.textContent = '⏳ Chargement Pyodide…';
  document.body.appendChild(statusEl);

  try {{
    const pyodide = await loadPyodide({{
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/'
    }});
    await pyodide.loadPackage(['micropip']);
    const micropip = pyodide.pyimport('micropip');
    await micropip.install(['dash', 'plotly']);

    statusEl.textContent = '✓ Pyodide prêt';
    statusEl.style.color = '#22c55e';
    setTimeout(() => statusEl.remove(), 3000);

    /* Inject le code de l'app */
    await pyodide.runPythonAsync(`
{app_code}
    `);
    console.log('[dash2html] Pyodide callbacks actifs');
  }} catch(e) {{
    statusEl.textContent = '⚠ Pyodide : ' + e.message;
    statusEl.style.color = '#ef4444';
    console.error('[dash2html] Pyodide erreur :', e);
  }}
}})();
</script>
<script src="https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js"></script>
"""


def inject_pyodide(soup: BeautifulSoup, app_path: str) -> BeautifulSoup:
    """Injecte Pyodide pour les callbacks dynamiques non pré-calculables."""
    try:
        app_code = Path(app_path).read_text(encoding="utf-8")
    except Exception:
        log.warning("  ⚠  Impossible de lire le code source pour Pyodide.")
        return soup

    script_html = PYODIDE_TEMPLATE.format(app_code=app_code.replace("`", "\\`"))
    body = soup.find("body")
    if body:
        body.append(BeautifulSoup(script_html, "html.parser"))
    log.info("  ✓  Fallback Pyodide injecté")
    return soup


# ── Meta Tags ──────────────────────────────────────────────────────────────────

def add_meta_tags(soup: BeautifulSoup, app_path: str):
    """Ajoute des meta tags utiles au <head>."""
    head = soup.find("head")
    if not head:
        return
    metas = [
        {"charset": "UTF-8"},
        {"name": "viewport",  "content": "width=device-width, initial-scale=1"},
        {"name": "generator", "content": "dash2html — https://github.com/your-repo"},
        {"name": "description","content": f"Export HTML autonome de {Path(app_path).name}"},
    ]
    for attrs in metas:
        tag = soup.new_tag("meta")
        for k, v in attrs.items():
            tag[k] = v
        head.insert(0, tag)


# ── Main Converter ─────────────────────────────────────────────────────────────

def convert(
    app_path:     str,
    output_path:  str | None = None,
    port:         int  = 8051,
    max_combos:   int  = 300,
    use_pyodide:  bool = False,
    no_precompute:bool = False,
    timeout:      int  = 30,
) -> str:
    """
    Pipeline complet de conversion.
    Retourne le chemin du fichier HTML généré.
    """
    app_path = os.path.abspath(app_path)
    if not os.path.isfile(app_path):
        raise FileNotFoundError(f"Fichier introuvable : {app_path}")

    if output_path is None:
        output_path = str(Path(app_path).with_suffix(".html"))

    log.info(f"")
    log.info(f"╔══════════════════════════════════════════╗")
    log.info(f"║       dash2html — Démarrage              ║")
    log.info(f"╚══════════════════════════════════════════╝")
    log.info(f"  Source  : {app_path}")
    log.info(f"  Sortie  : {output_path}")
    log.info(f"  Port    : {port}")
    log.info(f"")

    # 1. Charge l'app
    log.info("① Chargement de l'app Dash …")
    module, app = load_dash_app(app_path)

    # 2. Démarre le serveur
    log.info(f"② Démarrage du serveur sur le port {port} …")
    start_server(app, port)
    if not wait_for_server(port, timeout):
        raise RuntimeError(
            f"Le serveur Dash n'a pas répondu en {timeout}s sur le port {port}. "
            "Vérifiez que le port est disponible (--port XXXX)."
        )
    log.info(f"  ✓  Serveur opérationnel")

    base_url = f"http://127.0.0.1:{port}"
    session  = requests.Session()

    # 3. Récupère le HTML initial (après rendu React côté client si possible)
    log.info("③ Récupération du HTML initial …")
    try:
        r = session.get(base_url + "/", timeout=15)
        raw_html = r.text
    except Exception as e:
        raise RuntimeError(f"Impossible de récupérer le HTML : {e}")

    soup = BeautifulSoup(raw_html, "lxml")
    add_meta_tags(soup, app_path)

    # 4. Pré-calcule les callbacks
    lookup = {}
    if not no_precompute:
        log.info("④ Pré-calcul des callbacks …")
        precomputer = CallbackPrecomputer(base_url, max_combos=max_combos)
        try:
            lookup = precomputer.precompute()
        except Exception as e:
            log.warning(f"  ⚠  Pré-calcul partiel : {e}")

    # 5. Inline tous les assets
    log.info("⑤ Inline des assets (CSS / JS / images) …")
    inliner = AssetInliner(base_url)
    soup = inliner.inline_all(soup)

    # 6. Injecte le moteur de callbacks
    log.info("⑥ Injection du moteur de callbacks JS …")
    soup = inject_callback_engine(soup, lookup)

    # 7. Pyodide fallback (optionnel)
    if use_pyodide:
        log.info("⑦ Injection du fallback Pyodide …")
        soup = inject_pyodide(soup, app_path)

    # 8. Bannière offline
    soup = inject_banner(soup)

    # 9. Sauvegarde
    log.info(f"⑧ Sauvegarde → {output_path}")
    html_out = str(soup)
    Path(output_path).write_text(html_out, encoding="utf-8")

    size_kb = Path(output_path).stat().st_size / 1024
    log.info(f"")
    log.info(f"╔══════════════════════════════════════════╗")
    log.info(f"║  ✅  Conversion réussie                   ║")
    log.info(f"╚══════════════════════════════════════════╝")
    log.info(f"  Fichier : {output_path}")
    log.info(f"  Taille  : {size_kb:.1f} Ko")
    log.info(f"  Callbacks pré-calculés : {len(lookup)}")
    log.info(f"")

    return output_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convertit une app Dash en HTML portable autonome.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python dash_to_html.py app.py
  python dash_to_html.py app.py -o dashboard.html
  python dash_to_html.py app.py --port 8052 --max-combos 500
  python dash_to_html.py app.py --pyodide
  python dash_to_html.py app.py --no-precompute   # Juste inliner les assets
  python dash_to_html.py app.py --verbose
        """,
    )
    parser.add_argument(
        "app_path",
        help="Chemin vers le fichier Python de l'app Dash (ex: app.py)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Fichier HTML de sortie (défaut: même nom que l'app, extension .html)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8051,
        help="Port local pour le serveur Dash temporaire (défaut: 8051)",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=300,
        help="Nombre max de combinaisons d'inputs à pré-calculer par callback (défaut: 300)",
    )
    parser.add_argument(
        "--pyodide",
        action="store_true",
        help="Injecte Pyodide pour les callbacks complexes non pré-calculables "
             "(nécessite une connexion Internet à l'ouverture du HTML)",
    )
    parser.add_argument(
        "--no-precompute",
        action="store_true",
        help="Désactive le pré-calcul des callbacks (inline assets seulement)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Timeout (secondes) pour le démarrage du serveur Dash (défaut: 30)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mode verbeux (affiche les logs DEBUG)",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        convert(
            app_path     = args.app_path,
            output_path  = args.output,
            port         = args.port,
            max_combos   = args.max_combos,
            use_pyodide  = args.pyodide,
            no_precompute= args.no_precompute,
            timeout      = args.timeout,
        )
    except FileNotFoundError as e:
        log.error(f"❌  {e}")
        sys.exit(1)
    except RuntimeError as e:
        log.error(f"❌  {e}")
        sys.exit(2)
    except KeyboardInterrupt:
        log.info("Interruption utilisateur.")
        sys.exit(0)
    except Exception:
        log.error("❌  Erreur inattendue :")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
