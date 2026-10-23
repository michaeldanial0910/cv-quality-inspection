"""(Used to make demo/inspection_demo.mp4.) Drive the real Streamlit app in a visible Chromium on Xvfb and log timestamps of each scene.
Run under:  xvfb-run -s "-screen 0 1920x1080x24" python record.py  (ffmpeg x11grab started by this script)
"""
import json, os, subprocess, sys, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
URL = "http://localhost:8501"
marks = []
T0 = None

def mark(name):
    marks.append({"name": name, "t": round(time.time() - T0, 2)})
    print(f"{marks[-1]['t']:7.2f}  {name}", flush=True)

CURSOR_JS = """
(() => {
  if (window.__cursor) return;
  const c = document.createElement('div');
  c.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2 L4 19 L8.6 14.8 L11.6 21.4 L14.4 20.2 L11.4 13.6 L17.6 13.4 Z"
     fill="#1C2333" stroke="white" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
  Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex: 2147483647, pointerEvents:'none',
                          transform:'translate(-4px,-2px)', transition:'transform 60ms'});
  const ring = document.createElement('div');
  Object.assign(ring.style, {position:'fixed', width:'34px', height:'34px', borderRadius:'50%', border:'2px solid #2A6FD6',
     zIndex: 2147483646, pointerEvents:'none', opacity:'0', transform:'translate(-17px,-17px) scale(0.4)',
     transition:'opacity 350ms, transform 350ms'});
  const add = () => { document.body.appendChild(ring); document.body.appendChild(c); };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX+'px'; c.style.top = e.clientY+'px';
                                               ring.style.left = e.clientX+'px'; ring.style.top = e.clientY+'px'; }, true);
  document.addEventListener('mousedown', () => { ring.style.transition='none'; ring.style.opacity='0.9';
      ring.style.transform='translate(-17px,-17px) scale(0.4)'; requestAnimationFrame(() => { ring.style.transition='opacity 450ms, transform 450ms';
      ring.style.opacity='0'; ring.style.transform='translate(-17px,-17px) scale(1.3)'; }); }, true);
  window.__cursor = true;
})();
"""

def main():
    global T0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "30",
                           "-video_size", "1920x1080", "-draw_mouse", "0", "-i", os.environ["DISPLAY"],
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
                           f"{OUT}/raw.mp4"], stdin=subprocess.PIPE)
    T0 = time.time()
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context("/tmp/cv-demo-profile", executable_path="/opt/pw-browsers/chromium",
                              headless=False, no_viewport=True, ignore_default_args=["--enable-automation"],
                              args=["--kiosk", "--window-position=0,0", "--window-size=1920,1080",
                                    "--force-device-scale-factor=1.25", "--hide-scrollbars",
                                    "--disable-infobars", "--no-first-run"])
        ctx.add_init_script(CURSOR_JS)
        pg = ctx.pages[0] if ctx.pages else ctx.new_page()
        pg.goto(URL)
        pg.wait_for_selector("text=Should this casting ship?", timeout=60000)
        pg.wait_for_selector(".verdict", timeout=60000)
        pg.wait_for_timeout(2500)          # fonts settle
        pg.evaluate(CURSOR_JS)
        m = pg.mouse
        m.move(760, 500)
        mark("ready")
        main_el = "section.stMain, [data-testid='stMain']"

        def glide(locator, steps=28, click=True, dx=0, dy=0):
            box = locator.bounding_box()
            x, y = box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy
            m.move(x, y, steps=steps)
            pg.wait_for_timeout(180)
            if click:
                m.click(x, y)

        def scroll_to(px, dur_ms=1600):
            pg.evaluate(f"""(() => {{ const el = document.querySelector("{main_el}") || document.scrollingElement;
               const start = el.scrollTop, end = {px}, t0 = performance.now(), d = {dur_ms};
               function step(t) {{ const k = Math.min(1, (t - t0) / d), e = k<.5 ? 2*k*k : 1-Math.pow(-2*k+2,2)/2;
                  el.scrollTop = start + (end - start) * e; if (k < 1) requestAnimationFrame(step); }}
               requestAnimationFrame(step); }})()""")
            pg.wait_for_timeout(dur_ms + 150)

        def settle(sel=None, ms=900):
            if sel:
                pg.wait_for_selector(sel, timeout=60000)
            pg.wait_for_timeout(ms)

        def drag(key_cls, frac, steps=34):
            """Drag a slider thumb to a fraction of its track (one rerun on release, like a person would)."""
            thumb = pg.locator(f".{key_cls} [data-testid='stSliderThumbValue']").bounding_box()
            track = pg.locator(f".{key_cls} [role='group']").bounding_box()
            x0, y = thumb["x"] + thumb["width"] / 2, thumb["y"] + thumb["height"] + 12
            m.move(x0, y, steps=26)
            pg.wait_for_timeout(200)
            m.down()
            m.move(track["x"] + frac * track["width"], y, steps=steps)
            pg.wait_for_timeout(150)
            m.up()

        # ---- Inspect ----
        mark("inspect_start")
        pg.wait_for_timeout(1800)
        glide(pg.locator("[data-testid='stButtonGroup'] button", has_text="Clear pass"))
        settle("text=Auto-pass", 1500)
        mark("inspect_pass")
        glide(pg.locator("[data-testid='stButtonGroup'] button", has_text="Borderline"))
        settle("text=Send to a person", 1700)
        mark("inspect_border")
        glide(pg.get_by_text("Simulate camera drift (lighting, focus)"))
        settle("[data-testid='stSlider']", 600)
        drag("st-key-light", 0.10)
        settle("text=You changed the photo", 1600)
        mark("inspect_end")

        # ---- Results ----
        glide(pg.get_by_role("link", name="Results"))
        settle("text=Good parts thrown away", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1700)
        mark("results_start")
        scroll_to(330, 1300)
        pg.wait_for_timeout(1500)
        scroll_to(1240, 1500)
        pg.wait_for_timeout(500)
        drag("st-key-thr", 0.10, steps=40)
        settle(ms=600)
        pg.wait_for_timeout(1300)
        mark("results_end")

        # ---- Review queue ----
        glide(pg.get_by_role("link", name="Review queue"))
        settle(".tile", 1800)
        mark("review_start")
        scroll_to(260, 1200)
        pg.wait_for_timeout(600)
        glide(pg.get_by_text("Only the defective ones", exact=True))
        settle(ms=1800)
        mark("review_end")

        # ---- Limits ----
        glide(pg.get_by_role("link", name="Limits"))
        settle("text=It fails safe", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1300)
        mark("limits_start")
        scroll_to(300, 1300)
        pg.wait_for_timeout(1900)
        mark("limits_end")

        # ---- The data ----
        glide(pg.get_by_role("link", name="The data"))
        settle("text=The ready-made split", 300)
        pg.wait_for_selector(".vega-embed", timeout=60000)
        pg.wait_for_timeout(1300)
        mark("data_start")
        scroll_to(250, 1200)
        pg.wait_for_timeout(1800)
        mark("data_end")
        ctx.close()
    ff.communicate(b"q")
    json.dump(marks, open(f"{OUT}/marks.json", "w"), indent=1)

main()
