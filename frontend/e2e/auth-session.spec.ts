import { expect, test } from "@playwright/test";

const user = {
  id: "user-1",
  username: "doctor",
  role: "user",
  must_change_password: false,
  display_name: "Врач",
  organization: null,
  position: null,
  notes: null,
  profile_fields: {},
  allowed_profile_fields: [],
  active_sessions: 1,
};

test("cookie session survives reload and is cleared on logout", async ({ page, context }) => {
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const cookie = request.headers().cookie ?? "";

    if (path === "/api/auth/login") {
      await route.fulfill({
        contentType: "application/json",
        headers: {
          "set-cookie":
            "medassist_session=e2e-token; HttpOnly; SameSite=Strict; Path=/api; Max-Age=2592000",
        },
        body: JSON.stringify({ user }),
      });
      return;
    }
    if (path === "/api/auth/logout") {
      await route.fulfill({
        contentType: "application/json",
        headers: {
          "set-cookie":
            "medassist_session=; HttpOnly; SameSite=Strict; Path=/api; Max-Age=0",
        },
        body: JSON.stringify({ ok: true }),
      });
      return;
    }
    if (path === "/api/auth/me") {
      await route.fulfill({
        status: cookie.includes("medassist_session=e2e-token") ? 200 : 401,
        contentType: "application/json",
        body: JSON.stringify(
          cookie.includes("medassist_session=e2e-token")
            ? user
            : { error: "Требуется вход в систему" },
        ),
      });
      return;
    }
    if (path === "/api/events") {
      await route.fulfill({
        status: 200,
        headers: { "content-type": "text/event-stream" },
        body: "",
      });
      return;
    }
    if (path === "/api/config") {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ upload_max_body_size_mb: 50 }),
      });
      return;
    }
    if (path === "/api/calculators" && request.method() === "GET") {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          version: "2026.1",
          count: 1,
          groups: [
            {
              id: "cardiology",
              title: "Кардиология",
              calculators: [
                {
                  id: "mean-arterial-pressure",
                  group: "cardiology",
                  title: "Среднее артериальное давление",
                  description: "Расчёт MAP",
                  applicability: "При стабильном ритме.",
                  warnings: [],
                  reference: "MAP = (САД + 2 × ДАД) / 3",
                  version: "2026.1",
                  fields: [
                    { id: "systolic_bp", type: "number", label: "САД", min: 30, max: 300 },
                    { id: "diastolic_bp", type: "number", label: "ДАД", min: 10, max: 200 },
                  ],
                },
              ],
            },
          ],
        }),
      });
      return;
    }
    if (path === "/api/calculators/mean-arterial-pressure") {
      await route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({
          value: 80,
          unit: "мм рт. ст.",
          interpretation: "Расчётное среднее давление",
          details: [],
          warnings: [],
          reference: "MAP formula",
        }),
      });
      return;
    }
    await route.fulfill({ contentType: "application/json", body: "[]" });
  });

  await page.goto("/");
  await page.getByLabel("Логин").fill("doctor");
  await page.getByLabel("Пароль").fill("password");
  await page.getByRole("button", { name: "Войти" }).click();
  await expect(page.getByRole("heading", { name: "Документы" })).toBeVisible();

  expect(await page.evaluate(() => localStorage.getItem("clinical_ai_token"))).toBeNull();
  expect((await context.cookies()).find((cookie) => cookie.name === "medassist_session")?.httpOnly)
    .toBe(true);

  await page.reload();
  await expect(page.getByRole("heading", { name: "Документы" })).toBeVisible();

  await page.getByRole("link", { name: "Калькуляторы" }).click();
  await page.getByRole("link", { name: /Кардиология/ }).click();
  await page.getByLabel("САД").fill("120");
  await page.getByLabel("ДАД").fill("60");
  await page.getByRole("button", { name: "Рассчитать" }).click();
  await expect(page.getByText(/80 мм рт. ст./)).toBeVisible();

  await page.getByRole("button", { name: /doctor/ }).click();
  await page.getByRole("button", { name: "Выйти" }).click();
  await expect(page.getByText("Вход в МедАссистент")).toBeVisible();
});
