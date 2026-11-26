"""Exercise the native replay player from the main page, with page scripts disabled."""
import argparse
from pathlib import Path
import shutil
import time

from playwright.sync_api import expect, sync_playwright


def wait_for_media(player, predicate, timeout=12):
    # Poll from Python so document JavaScript can remain disabled.
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if player.evaluate(predicate):
            return
        time.sleep(0.1)
    state = player.evaluate("v => [v.currentTime, v.duration, v.paused, v.error?.message]")
    raise AssertionError(f"Player did not reach {predicate}. State: {state}")


def check(browser, url, profile, device, screenshots):
    with browser.new_context(**device, java_script_enabled=False) as context:
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        link = page.get_by_role("link", name="저장 결과 영상 재생 · 21초", exact=False)
        expect(link).to_be_visible()
        link.click()
        player = page.locator("video[controls]")
        expect(player).to_be_visible()
        wait_for_media(player, "v => v.currentTime > 0 && !v.paused")
        assert player.evaluate("v => v.duration === 21 && v.playbackRate === 1")
        # Let the player reach the end without any seeks or playback-rate changes.
        wait_for_media(player, "v => v.ended", timeout=35)
        assert player.evaluate("v => v.played.length === 1 && v.played.start(0) < 0.05"
                               " && v.played.end(0) >= v.duration - 0.05")
        player.evaluate("v => v.play()")
        wait_for_media(player, "v => !v.paused && v.currentTime > 0.1 && v.currentTime < 2")
        player.evaluate("v => v.pause()")
        position = player.evaluate("v => v.currentTime")
        time.sleep(0.35)
        assert player.evaluate("v => v.paused && v.currentTime") == position
        for label, target in (("low-light", 13), ("summary-first", 18.02), ("summary-last", 20.96)):
            player.evaluate("(v, t) => { v.currentTime = t }", target)
            wait_for_media(player, f"v => !v.seeking && Math.abs(v.currentTime - {target}) < 0.02")
            if screenshots:
                screenshots.mkdir(parents=True, exist_ok=True)
                player.screenshot(path=screenshots / f"{profile}-{label}.png")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        # Some engines consider a paused seek onto the last frame already ended.
        player.evaluate("v => { v.currentTime = 19 }")
        wait_for_media(player, "v => !v.seeking && Math.abs(v.currentTime - 19) < 0.02")
        player.evaluate("v => v.play()")
        wait_for_media(player, "v => v.ended")
        page.get_by_role("link", name="장면 비교로 돌아가기").click()
        expect(link).to_be_visible()
        print(f"{browser.browser_type.name} {browser.version} {profile}: full 21s playback, "
              "restart, pause, low-light / summary seeks, resume, end and return passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--browser", choices=("chromium", "webkit"), default="chromium")
    parser.add_argument("--profile", choices=("desktop", "mobile", "both"), default="both")
    parser.add_argument("--screenshots", type=Path)
    args = parser.parse_args()
    with sync_playwright() as playwright:
        engine = getattr(playwright, args.browser)
        launch = {"executable_path": shutil.which("chromium")} if args.browser == "chromium" else {}
        devices = {"desktop": {"viewport": {"width": 1365, "height": 850}},
                   "mobile": playwright.devices["iPhone 13" if args.browser == "webkit" else "Pixel 7"]}
        with engine.launch(**launch) as browser:
            for name, device in devices.items():
                if args.profile in ("both", name):
                    check(browser, args.url, name, device, args.screenshots)
