import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { describe, expect, it } from "vitest";
import { AuthProvider, useAuth, type AuthUser } from "@/hooks/useAuth";
import { mockServer } from "@/test/mocks/server";

const user: AuthUser = {
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

function AuthProbe() {
  const auth = useAuth();
  return (
    <div>
      <span>{auth.isLoading ? "loading" : auth.user?.username ?? "anonymous"}</span>
      <button type="button" onClick={() => void auth.login("doctor", "password")}>
        login
      </button>
      <button type="button" onClick={() => void auth.refreshUser()}>
        refresh
      </button>
    </div>
  );
}

function renderAuth() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <AuthProbe />
      </AuthProvider>
    </QueryClientProvider>,
  );
}

describe("AuthProvider cookie session flow", () => {
  it("bootstraps anonymously after /me returns 401", async () => {
    mockServer.use(
      http.get("/api/auth/me", () =>
        HttpResponse.json({ error: "Требуется вход" }, { status: 401 }),
      ),
    );

    renderAuth();

    expect(await screen.findByText("anonymous")).toBeInTheDocument();
  });

  it("logs in without writing a bearer token to localStorage", async () => {
    mockServer.use(
      http.get("/api/auth/me", () =>
        HttpResponse.json({ error: "Требуется вход" }, { status: 401 }),
      ),
      http.post("/api/auth/login", () => HttpResponse.json({ user })),
    );

    renderAuth();
    await screen.findByText("anonymous");
    fireEvent.click(screen.getByRole("button", { name: "login" }));

    await waitFor(() => expect(screen.getByText("doctor")).toBeInTheDocument());
    expect(localStorage.getItem("clinical_ai_token")).toBeNull();
  });

  it("keeps the current user on a transient /me failure", async () => {
    let meStatus = 401;
    mockServer.use(
      http.get("/api/auth/me", () =>
        HttpResponse.json(
          meStatus === 200 ? user : { error: "Сервис временно недоступен" },
          { status: meStatus },
        ),
      ),
      http.post("/api/auth/login", () => HttpResponse.json({ user })),
    );

    renderAuth();
    await screen.findByText("anonymous");
    fireEvent.click(screen.getByRole("button", { name: "login" }));
    await screen.findByText("doctor");

    meStatus = 500;
    fireEvent.click(screen.getByRole("button", { name: "refresh" }));
    await waitFor(() => expect(screen.getByText("doctor")).toBeInTheDocument());
  });
});
