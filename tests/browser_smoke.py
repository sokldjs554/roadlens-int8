"""Exercise the published RoadLens demo using Chromium and the Playwright Python package."""

import argparse
import math
from pathlib import Path
import shutil
from urllib.parse import urljoin, urlsplit

from playwright.sync_api import expect, sync_playwright


def check_view(browser, url, width, height, screenshots, check_links):
    mobile = width <= 740
    context = browser.new_context(viewport={"width": width, "height": height},
                                  is_mobile=mobile, has_touch=mobile, device_scale_factor=1)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("requestfailed", lambda req: errors.append(req.url + " " + str(req.failure))
            if req.failure != "net::ERR_ABORTED" else None)
    try:
        page.goto(url, wait_until="networkidle")
        expect(page.locator("#dashboard")).to_be_visible()
        data_response = context.request.get(urljoin(page.url, "results.json"))
        assert data_response.ok
        data = data_response.json()
        scenes = data["scenes"]
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        expect(page.locator("#group-rows tr")).to_have_count(len(data["groups"]))
        expect(page.locator("#gallery button")).to_have_count(len(scenes))
        expect(page.locator("#modes button")).to_have_count(4)
        expect(page.locator("#metrics tr")).to_have_count(6)
        summary = page.locator(".glance" if mobile else ".summary")
        assert summary.bounding_box()["y"] + summary.bounding_box()["height"] <= height
        for model in data["models"].values():
            expect(summary).to_contain_text(f'{model["iou"] * 100:.2f}%')
            expect(summary).to_contain_text(f'{model["accuracy"] * 100:.2f}%')
            expect(summary).to_contain_text(f'{model["p50_ms"]:.3f}')
            expect(summary).to_contain_text(f'{model["bytes"] / 1024:.1f}')
        if screenshots:
            screenshots.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=screenshots / f"roadlens-{width}.png", full_page=True)
        expect(page.locator("#prev")).to_be_disabled()
        for index, scene in enumerate(scenes):
            page.locator("#scene-menu").select_option(str(index))
            expect(page.locator("#scene-stats")).to_contain_text(scene["id"])
            expect(page.locator("#scene-stats")).to_contain_text(f'{scene["iou_int8"] * 100:.2f}%')
            for view_index, view in enumerate(("input", "truth", "fp32", "int8")):
                button = page.locator("#modes button").nth(view_index)
                button.click()
                expect(button).to_have_attribute("aria-pressed", "true")
                expect(page.locator("#view-base")).to_have_attribute("src", scene[view])
                page.wait_for_function("document.querySelector('#view-base').naturalWidth > 0")
            page.locator("#split-button").click()
            expect(page.locator("#view-top")).to_have_attribute("src", scene["fp32"])
            expect(page.locator("#view-base")).to_have_attribute("src", scene["int8"])
        expect(page.locator("#next")).to_be_disabled()
        slider = page.locator("#wipe")
        slider.focus()
        slider.press("Home")
        slider.press("ArrowRight")
        expect(slider).to_have_value("1")
        expect(slider).to_have_attribute("aria-valuetext", "왼쪽 FP32 1%, 오른쪽 INT8 99%")
        slider.press("End")
        expect(slider).to_have_value("100")
        rect = slider.bounding_box()
        page.mouse.click(rect["x"] + rect["width"] / 3, rect["y"] + rect["height"] / 2)
        assert math.isclose(float(slider.input_value()), 33, abs_tol=2)
        page.locator("#gallery button").first.click()
        expect(page.locator("#scene-menu")).to_have_value("0")
        page.locator("#next").click()
        expect(page.locator("#scene-menu")).to_have_value("1")
        page.locator("#prev").click()
        expect(page.locator("#scene-menu")).to_have_value("0")
        if check_links:
            links = page.locator("a[href]").evaluate_all(
                "anchors => anchors.map(a => a.getAttribute('href'))")
            paths = {s[key] for s in scenes for key in ("input", "truth", "fp32", "int8")}
            for link in links:
                if not link.startswith("#") and not urlsplit(link).netloc:
                    paths.add(link)
            for path in sorted(paths):
                response = context.request.get(urljoin(page.url, path))
                assert response.ok and response.body(), (path, response.status)
                if path.endswith(".jpg"):
                    assert response.body().startswith(b"\xff\xd8"), path
                if path.endswith(".zip"):
                    assert response.body().startswith(b"PK"), path
                response.dispose()
        assert not errors, errors
        print(f"{width}×{height}: {len(scenes)} scenes, four modes, slider, metrics and navigation passed")
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Demo URL including any deployment subdirectory and trailing slash")
    parser.add_argument("--chromium", default=shutil.which("chromium") or shutil.which("google-chrome"))
    parser.add_argument("--screenshots", type=Path)
    parser.add_argument("--skip-linked-files", action="store_true", help="For UI-only local testing")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.chromium, args=["--no-sandbox"])
        try:
            for width, height in ((1365, 850), (360, 760)):
                check_view(browser, args.url, width, height, args.screenshots,
                           not args.skip_linked_files and width == 1365)
        finally:
            browser.close()


if __name__ == "__main__":
    main()
