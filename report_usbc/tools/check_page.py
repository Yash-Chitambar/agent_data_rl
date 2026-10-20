#!/usr/bin/env python3
"""Smoke-test a project page in headless Chromium.

Serves the site folder over HTTP (module scripts and fetch() do not work from file://),
scrolls every <figure class="viz"> into view, waits for it to mount, and reports:
  - console errors and failed requests
  - figures that rendered an error or never mounted
  - one screenshot per figure + a full page, in light and dark mode, desktop and phone

    pip install playwright && playwright install chromium
    python tools/check_page.py path/to/site [--out shots/] [--wait 2.5]

Exit code 1 if anything failed, so a harness can loop until it is clean.
"""
import argparse
import functools
import http.server
import sys
import threading
from pathlib import Path


def serve(root):
    handler = functools.partial(QuietHandler, directory=str(root))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm",
                      ".json": "application/json", ".svg": "image/svg+xml"}

    def log_message(self, *a):
        pass


def mirror_routes(ctx, node_modules):
    """Serve cdn.jsdelivr.net/npm/<pkg>@<ver>/<path> from a local node_modules (offline testing)."""
    nm = Path(node_modules)

    def handle(route):
        rest = route.request.url.split("/npm/", 1)[1].split("?")[0]
        if rest.startswith("@"):
            scope, pkgver, *path = rest.split("/")
            pkg = f"{scope}/{pkgver.rsplit('@', 1)[0]}"
        else:
            pkgver, *path = rest.split("/")
            pkg = pkgver.rsplit("@", 1)[0]
        f = nm / pkg / "/".join(path)
        if f.is_file():
            ctype = QuietHandler.extensions_map.get(f.suffix, "application/octet-stream")
            if f.suffix == ".css":
                ctype = "text/css"
            route.fulfill(path=str(f), headers={"Content-Type": ctype, "Access-Control-Allow-Origin": "*"})
        else:
            route.fulfill(status=404, body=f"not in mirror: {pkg}/{'/'.join(path)}")

    ctx.route("https://cdn.jsdelivr.net/npm/**", handle)
    # Web fonts are cosmetic: skip them offline instead of failing.
    ctx.route(lambda u: "fonts.googleapis.com" in u or "fonts.gstatic.com" in u,
              lambda r: r.fulfill(status=200, body="", headers={"Content-Type": "text/css"}))


def run(site, out, wait, npm_mirror=None, page_name=""):
    from playwright.sync_api import sync_playwright

    httpd, url = serve(site)
    out.mkdir(parents=True, exist_ok=True)
    problems = []
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        for scheme in ("light", "dark"):
            for label, vp in (("desktop", {"width": 1280, "height": 900}), ("phone", {"width": 390, "height": 844})):
                ctx = browser.new_context(viewport=vp, color_scheme=scheme, device_scale_factor=1)
                if npm_mirror:
                    mirror_routes(ctx, npm_mirror)
                page = ctx.new_page()
                tag = f"{scheme}-{label}"
                page.on("console", lambda m, tag=tag: m.type == "error" and problems.append(f"[{tag}] console: {m.text}"))
                page.on("pageerror", lambda e, tag=tag: problems.append(f"[{tag}] page error: {e}"))
                page.on("requestfailed", lambda r, tag=tag: problems.append(f"[{tag}] request failed: {r.url} ({r.failure})"))
                page.on("response", lambda r, tag=tag: r.status >= 400 and problems.append(f"[{tag}] HTTP {r.status}: {r.url}"))
                page.goto(url + page_name, wait_until="networkidle")
                figs = page.locator("figure.viz")
                n = figs.count()
                for i in range(n):
                    fig = figs.nth(i)
                    name = fig.get_attribute("data-fig")
                    fig.scroll_into_view_if_needed()
                    try:
                        page.wait_for_function("(el) => el.dataset.mounted === '1'", arg=fig.element_handle(), timeout=20000)
                    except Exception:
                        problems.append(f"[{tag}] figure '{name}' never finished mounting")
                    page.wait_for_timeout(int(wait * 1000))
                    err = fig.locator(".viz-error")
                    if err.count():
                        problems.append(f"[{tag}] figure '{name}': {err.first.inner_text()}")
                    if scheme == "light" or label == "desktop":
                        fig.screenshot(path=str(out / f"{tag}-{i:02d}-{name}.png"))
                # Horizontal overflow breaks phone layouts silently.
                overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                if overflow > 1:
                    problems.append(f"[{tag}] page scrolls horizontally by {overflow}px")
                page.screenshot(path=str(out / f"{tag}-page.png"), full_page=True)
                print(f"{tag}: checked {n} figures")
                ctx.close()
        browser.close()
    httpd.shutdown()
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("site", type=Path)
    ap.add_argument("--out", type=Path, default=Path("shots"))
    ap.add_argument("--wait", type=float, default=2.5, help="seconds to let animations/sims run per figure")
    ap.add_argument("--npm-mirror", type=Path, default=None,
                    help="node_modules folder to serve cdn.jsdelivr.net/npm URLs from (offline testing: npm i three@0.169.0 katex@0.16.9)")
    ap.add_argument("--page", default="", help="page path under the site (default index.html)")
    a = ap.parse_args()
    probs = run(a.site.resolve(), a.out, a.wait, a.npm_mirror, a.page)
    if probs:
        print("\nPROBLEMS:")
        for x in dict.fromkeys(probs):
            print("  -", x)
        sys.exit(1)
    print(f"\nOK, screenshots in {a.out}/")
