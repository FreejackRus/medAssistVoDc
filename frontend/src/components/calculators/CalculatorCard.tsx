import { useState } from "react";
import { Calculator, Copy, Check } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import {
  useCalculate,
  type CalculatorConfig,
  type CalculatorResult,
  type FieldConfig,
} from "@/hooks/useCalculators";

interface Props {
  config: CalculatorConfig;
}

function getDefaultValues(fields: FieldConfig[]): Record<string, string | boolean> {
  const values: Record<string, string | boolean> = {};
  for (const f of fields) {
    if (f.type === "checkbox") {
      values[f.name] = (f.defaultValue as boolean) ?? false;
    } else {
      values[f.name] = String(f.defaultValue ?? "");
    }
  }
  return values;
}

function parseValues(fields: FieldConfig[], values: Record<string, string | boolean>) {
  const parsed: Record<string, unknown> = {};
  for (const f of fields) {
    if (f.type === "checkbox") {
      parsed[f.name] = values[f.name];
    } else if (f.type === "number") {
      parsed[f.name] = Number(values[f.name]);
    } else {
      parsed[f.name] = values[f.name];
    }
  }
  return parsed;
}

function validateFields(fields: FieldConfig[], values: Record<string, string | boolean>): string | null {
  for (const f of fields) {
    if (f.type !== "number") continue;
    const raw = String(values[f.name] ?? "");
    if (!raw.trim()) return `Заполните поле «${f.label}»`;
    const num = Number(raw);
    if (isNaN(num)) return `«${f.label}»: введите число`;
    if (f.min != null && num < f.min) return `«${f.label}»: минимум ${f.min}`;
    if (f.max != null && num > f.max) return `«${f.label}»: максимум ${f.max}`;
  }
  return null;
}

export default function CalculatorCard({ config }: Props) {
  const [values, setValues] = useState(() => getDefaultValues(config.fields));
  const [result, setResult] = useState<CalculatorResult | null>(null);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const calculate = useCalculate(config.endpoint);

  const handleSubmit = async () => {
    const err = validateFields(config.fields, values);
    if (err) {
      setValidationError(err);
      return;
    }
    setValidationError(null);
    const parsed = parseValues(config.fields, values);
    const res = await calculate.mutateAsync(parsed);
    setResult(res);
  };

  const setValue = (name: string, value: string | boolean) => {
    setValues((prev) => ({ ...prev, [name]: value }));
    setResult(null);
    setValidationError(null);
  };

  const copyResult = () => {
    if (!result) return;
    navigator.clipboard.writeText(
      `${result.result} ${result.unit} — ${result.interpretation}`,
    );
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <Card className="overflow-visible">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <Calculator className="h-4 w-4" />
          {config.title}
        </CardTitle>
        <p className="text-xs text-muted-foreground">{config.description}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        {config.fields.map((field) => {
          const fieldId = `${config.id}-${field.name}`;

          return (
            <div key={field.name}>
              <label className="block text-sm font-medium" htmlFor={fieldId}>
                {field.label}
              </label>
              {field.type === "select" ? (
                <Select
                  id={fieldId}
                  className="mt-1"
                  value={values[field.name] as string}
                  options={field.options ?? []}
                  onValueChange={(value) => setValue(field.name, value)}
                />
              ) : field.type === "checkbox" ? (
                <label className="mt-2 flex h-6 w-fit cursor-pointer items-center gap-2.5 text-sm leading-none text-foreground">
                  <Checkbox
                    id={fieldId}
                    checked={values[field.name] as boolean}
                    onChange={(e) => setValue(field.name, e.target.checked)}
                  />
                  Да
                </label>
              ) : (
                <Input
                  id={fieldId}
                  type="number"
                  className="mt-1"
                  placeholder={field.placeholder}
                  min={field.min}
                  max={field.max}
                  step="any"
                  value={String(values[field.name] ?? "")}
                  onChange={(e) => setValue(field.name, e.target.value)}
                />
              )}
            </div>
          );
        })}

        <Button
          className="w-full"
          onClick={handleSubmit}
          disabled={calculate.isPending}
        >
          {calculate.isPending ? "Расчёт..." : "Рассчитать"}
        </Button>

        {(validationError || calculate.error) && (
          <p className="text-sm text-destructive">
            {validationError || calculate.error?.message}
          </p>
        )}

        {result && (
          <div className="flex items-start gap-2 rounded-lg bg-muted p-3">
            <div className="flex-1 space-y-1">
              <p className="text-lg font-semibold">
                {result.result} {result.unit}
              </p>
              <p className="text-sm text-muted-foreground">
                {result.interpretation}
              </p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="shrink-0 h-8 w-8"
              title="Копировать результат"
              onClick={copyResult}
            >
              {copied ? (
                <Check className="h-4 w-4 text-green-600" />
              ) : (
                <Copy className="h-4 w-4" />
              )}
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
