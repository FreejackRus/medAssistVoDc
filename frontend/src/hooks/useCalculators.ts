import { useMutation, useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface CalculatorResult {
  value: string | number;
  unit: string;
  interpretation: string;
  details: string[];
  warnings: string[];
  reference: string;
}

export interface CalculatorField {
  id: string;
  type: "number" | "select" | "checkbox";
  label: string;
  options?: Array<{ value: string; label: string }>;
  default?: string | number | boolean;
  min?: number;
  max?: number;
}

export interface CalculatorConfig {
  id: string;
  group: string;
  title: string;
  description: string;
  fields: CalculatorField[];
  warnings: string[];
  applicability: string;
  reference: string;
  version: string;
}

export interface CalculatorGroup {
  id: string;
  title: string;
  calculators: CalculatorConfig[];
}

export interface CalculatorRegistry {
  version: string;
  count: number;
  groups: CalculatorGroup[];
}

export function useCalculatorRegistry() {
  return useQuery<CalculatorRegistry>({
    queryKey: ["calculator-registry"],
    queryFn: () => apiFetch("/calculators"),
    staleTime: 24 * 60 * 60 * 1000,
  });
}

export function useCalculate(calculatorId: string) {
  return useMutation<CalculatorResult, Error, Record<string, unknown>>({
    mutationFn: (body) =>
      apiFetch(`/calculators/${calculatorId}`, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  });
}
