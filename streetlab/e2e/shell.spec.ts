import { expect, test, type Page } from '@playwright/test';

/**
 * The shell's spacing contract, asserted against real layout rather than the
 * stylesheet: every surface sits one gutter from its neighbour, and every run
 * of content inside a chrome surface shares one left inset. jsdom cannot see
 * any of this — the unit tests can only prove a collapsed panel unmounted, not
 * that the viewport actually claimed its track.
 */

/**
 * Read the shell's own spacing tokens rather than hard-coding them: they are
 * responsive, and the point of these tests is that everything agrees with the
 * scale at whatever width the run happens to use — not that the scale is any
 * particular number.
 */
async function scale(page: Page) {
  return page.evaluate(() => {
    const s = getComputedStyle(document.documentElement);
    return {
      gutter: parseFloat(s.getPropertyValue('--gutter')),
      pad: parseFloat(s.getPropertyValue('--pad-panel')),
    };
  });
}

const box = (page: Page, selector: string) =>
  page.locator(selector).first().boundingBox();

/** Left inset of an element measured from its containing surface. */
async function inset(page: Page, surface: string, child: string) {
  const s = await box(page, surface);
  const c = await box(page, child);
  if (!s || !c) throw new Error(`missing ${surface} or ${child}`);
  return Math.round(c.x - s.x);
}

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (err) => {
    throw new Error(`uncaught page error: ${err.message}`);
  });
  await page.goto('/?mock=1');
  await expect(page.locator('.viewport')).toBeVisible();
});

test('every surface sits one gutter from the viewport', async ({ page }) => {
  const { gutter: GUTTER } = await scale(page);
  const vp = (await box(page, '.viewport'))!;
  const sidebar = (await box(page, '.sidebar'))!;
  const panel = (await box(page, '.panel'))!;
  const toolbar = (await box(page, '.toolbar'))!;
  const card = (await box(page, '.tcard'))!;

  expect(Math.round(vp.x - (sidebar.x + sidebar.width))).toBe(GUTTER);
  expect(Math.round(panel.x - (vp.x + vp.width))).toBe(GUTTER);
  expect(Math.round(vp.y - (toolbar.y + toolbar.height))).toBe(GUTTER);
  expect(Math.round(card.y - (vp.y + vp.height))).toBe(GUTTER);

  // The telemetry strip spans the window, so its own gutters are to the edges.
  const size = page.viewportSize()!;
  const last = (await page.locator('.tcard').last().boundingBox())!;
  expect(Math.round(card.x)).toBe(GUTTER);
  expect(Math.round(size.width - (last.x + last.width))).toBe(GUTTER);
  expect(Math.round(size.height - (card.y + card.height))).toBe(GUTTER);
});

test('content in each chrome surface shares one left inset', async ({ page }) => {
  const { pad: PAD } = await scale(page);
  for (const child of [
    '.sidebar-title',
    '.location-search input',
    '.scenario',
    '.foot-btn',
  ]) {
    expect(await inset(page, '.sidebar', child), child).toBe(PAD);
  }

  // The inspector's own 1px left border offsets its content by that pixel.
  for (const child of ['.panel-tab', '.field-title', '.ctl-label']) {
    expect(await inset(page, '.panel', child), child).toBe(PAD + 1);
  }

  const toolbar = (await box(page, '.toolbar'))!;
  const brand = (await box(page, '.brand'))!;
  const right = (await box(page, '.toolbar-group--right'))!;
  expect(Math.round(brand.x - toolbar.x)).toBe(PAD);
  expect(Math.round(toolbar.x + toolbar.width - (right.x + right.width))).toBe(PAD);
});

test("no inspector control escapes the panel's text column", async ({ page }) => {
  const { pad: PAD } = await scale(page);
  const panel = (await box(page, '.panel'))!;
  const strays = await page.evaluate(
    ({ pad }) => {
      const p = document.querySelector('.panel')!.getBoundingClientRect();
      return [...document.querySelectorAll('.panel-body *')]
        .filter((e) => {
          const r = e.getBoundingClientRect();
          return (
            r.width > 0 &&
            (r.x < p.x + pad + 0.5 || r.right > p.right - pad + 0.5)
          );
        })
        .map((e) => e.className || e.tagName);
    },
    { pad: PAD },
  );
  expect(strays, `panel at ${panel.x}`).toEqual([]);
});

const PANELS = [
  { id: 'scenarios', surface: '.sidebar', axis: 'width' },
  { id: 'inspector', surface: '.panel', axis: 'width' },
  { id: 'telemetry', surface: '.telemetry', axis: 'height' },
] as const;

for (const { id, surface, axis } of PANELS) {
  test(`collapsing ${id} hands its space to the viewport`, async ({ page }) => {
    const before = (await box(page, '.viewport'))!;
    const folded = (await box(page, surface))!;

    await page.getByTestId(`toggle-${id}`).click();
    await expect(page.locator(surface)).toHaveCount(0);

    // The regression this exists for: the viewport must take the vacated
    // track, not slide into the collapsed one and end up zero-sized.
    const after = (await box(page, '.viewport'))!;
    expect(Math.round(after[axis])).toBe(Math.round(before[axis] + folded[axis]));
    expect(after.width).toBeGreaterThan(0);
    expect(after.height).toBeGreaterThan(0);

    // The handle outlives its panel and is the way back.
    await page.getByTestId(`toggle-${id}`).click();
    await expect(page.locator(surface)).toHaveCount(1);
    const restored = (await box(page, '.viewport'))!;
    expect(Math.round(restored[axis])).toBe(Math.round(before[axis]));
  });
}

test('all three collapsed leaves a full-bleed viewport and three live handles', async ({
  page,
}) => {
  const { gutter: GUTTER } = await scale(page);
  for (const { id } of PANELS) await page.getByTestId(`toggle-${id}`).click();

  const size = page.viewportSize()!;
  const vp = (await box(page, '.viewport'))!;
  const toolbar = (await box(page, '.toolbar'))!;
  expect(Math.round(vp.x)).toBe(GUTTER);
  expect(Math.round(size.width - (vp.x + vp.width))).toBe(GUTTER);
  expect(Math.round(size.height - (vp.y + vp.height))).toBe(GUTTER);
  expect(Math.round(vp.y - toolbar.height)).toBe(GUTTER);

  // Every handle stays on screen, or a folded surface would be unrecoverable.
  for (const { id } of PANELS) {
    const h = (await box(page, `[data-testid="toggle-${id}"]`))!;
    expect(h.x).toBeGreaterThanOrEqual(0);
    expect(h.y).toBeGreaterThanOrEqual(0);
    expect(h.x + h.width).toBeLessThanOrEqual(size.width);
    expect(h.y + h.height).toBeLessThanOrEqual(size.height);
  }
});

test('the inspector tabs fit their strip at both panel widths', async ({ page }) => {
  const overflows = async () =>
    page.evaluate(() => {
      const strip = document.querySelector('.panel-tabs')!;
      const clipped = [...document.querySelectorAll('.panel-tab span')].filter(
        (sp) => {
          if (getComputedStyle(sp).display === 'none') return false;
          const r = document.createRange();
          r.selectNodeContents(sp);
          return r.getBoundingClientRect().width > sp.getBoundingClientRect().width + 0.5;
        },
      ).length;
      return { strip: strip.scrollWidth - strip.clientWidth, clipped };
    });

  // Above the breakpoint, four labelled tabs share the strip; below it the
  // labels give way to icons. Neither may clip — this is what shipped broken,
  // with "Events" cut off by the panel's edge.
  for (const width of [1600, 1280]) {
    await page.setViewportSize({ width, height: 900 });
    await expect(page.locator('.panel-tab').first()).toBeVisible();
    expect(await overflows(), `at ${width}px`).toEqual({ strip: 0, clipped: 0 });
  }
});
