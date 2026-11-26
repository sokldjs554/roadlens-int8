"""Check the replay entry point and native playback with JavaScript disabled."""
import argparse
import shutil
import time
from playwright.sync_api import expect, sync_playwright


def wait_for_media(player, predicate):
    # Poll outside the page so this also works when document scripts are disabled.
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline:
        if player.evaluate(predicate):
            return
        time.sleep(0.1)
    raise AssertionError(f"Player did not reach {predicate}")


def check(browser, url, mobile):
    viewport = {"width": 360, "height": 760} if mobile else {"width": 1365, "height": 850}
    with browser.new_context(viewport=viewport, is_mobile=mobile, has_touch=mobile,
                             java_script_enabled=False) as context:
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded")
        link = page.get_by_role("link", name="저장 결과 영상 재생", exact=False)
        expect(link).to_be_visible(timeout=2500)
        link.click()
        player = page.locator("video[controls]")
        expect(player).to_be_visible()
        wait_for_media(player, "v => v.currentTime > 0.2")
        assert abs(player.evaluate("v => v.duration") - 18) < 0.1
        player.evaluate("v => v.pause()")
        assert player.evaluate("v => v.paused")
        player.evaluate("v => { v.currentTime = 12.2 }")
        wait_for_media(player, "v => !v.seeking")
        assert player.evaluate("v => v.currentTime") >= 12
        player.evaluate("v => v.play()")
        wait_for_media(player, "v => v.currentTime > 12.5")
        player.evaluate("v => { v.currentTime = v.duration - 0.2 }")
        wait_for_media(player, "v => v.ended")
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.get_by_role("link", name="장면 비교로 돌아가기").click()
        expect(link).to_be_visible()
        print(f"{viewport['width']}px: link, autoplay, pause, seek, resume, end and return passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        with playwright.chromium.launch(executable_path=shutil.which("chromium")) as browser:
            for mobile in (False, True):
                check(browser, args.url, mobile)
