import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { HttpResponse, http } from "msw";
import { describe, expect, it } from "vitest";
import CalculatorCard from "@/components/calculators/CalculatorCard";
import type { CalculatorConfig } from "@/hooks/useCalculators";
import { mockServer } from "@/test/mocks/server";

const config: CalculatorConfig = {
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
};

describe("CalculatorCard", () => {
  it("renders registry fields and extended backend result", async () => {
    mockServer.use(
      http.post("/api/calculators/mean-arterial-pressure", async ({ request }) => {
        expect(await request.json()).toEqual({ systolic_bp: 120, diastolic_bp: 60 });
        return HttpResponse.json({
          value: 80,
          unit: "мм рт. ст.",
          interpretation: "Расчётное среднее давление",
          details: ["САД 120", "ДАД 60"],
          warnings: [],
          reference: config.reference,
        });
      }),
    );
    const queryClient = new QueryClient();
    render(
      <QueryClientProvider client={queryClient}>
        <CalculatorCard config={config} />
      </QueryClientProvider>,
    );

    fireEvent.change(screen.getByLabelText("САД"), { target: { value: "120" } });
    fireEvent.change(screen.getByLabelText("ДАД"), { target: { value: "60" } });
    fireEvent.click(screen.getByRole("button", { name: "Рассчитать" }));

    expect(await screen.findByText(/80 мм рт. ст./)).toBeInTheDocument();
    expect(screen.getByText("Расчётное среднее давление")).toBeInTheDocument();
    expect(screen.getByText("САД 120")).toBeInTheDocument();
  });
});
