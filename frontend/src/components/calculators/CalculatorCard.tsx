import { useState } from "react";
import { AlertTriangle, BookOpen, Calculator, Copy, Check } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Select } from "@/components/ui/select";
import {
  useCalculate,
  type CalculatorConfig,
  type CalculatorField,
  type CalculatorResult,
} from "@/hooks/useCalculators";

interface Props {
  config: CalculatorConfig;
}

function getDefaultValues(fields: CalculatorField[]): Record<string, string | boolean> {
  const values: Record<string, string | boolean> = {};
  for (const f of fields) {
    if (f.type === "checkbox") {
      values[f.id] = (f.default as boolean) ?? false;
    } else {
      values[f.id] = String(f.default ?? "");
    }
  }
  return values;
}

function parseValues(fields: CalculatorField[], values: Record<string, string | boolean>) {
  const parsed: Record<string, unknown> = {};
  for (const f of fields) {
    if (f.type === "checkbox") {
      parsed[f.id] = values[f.id];
    } else if (f.type === "number") {
      parsed[f.id] = Number(values[f.id]);
    } else {
      parsed[f.id] = values[f.id];
    }
  }
  return parsed;
}

function validateFields(
  fields: CalculatorField[],
  values: Record<string, string | boolean>,
): string | null {
  for (const f of fields) {
    if (f.type !== "number") continue;
    const raw = String(values[f.id] ?? "");
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
  const calculate = useCalculate(config.id);

  const handleSubmit = async () => {
    const err = validateFields(config.fields, values);
    if (err) {
      setValidationError(err);
      return;
    }
    setValidationError(null);
    const parsed = parseValues(config.fields, values);
    try {
      const res = await calculate.mutateAsync(parsed);
      setResult(res);
    } catch {
      // The mutation exposes the localized error below the form.
    }
  };

  const setValue = (name: string, value: string | boolean) => {
    setValues((prev) => ({ ...prev, [name]: value }));
    setResult(null);
    setValidationError(null);
  };

  const copyResult = () => {
    if (!result) return;
    navigator.clipboard.writeText(
      `${result.value} ${result.unit} — ${result.interpretation}`,
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
        <div className="rounded-md border bg-muted/30 p-3 text-xs text-muted-foreground">
          <p>{config.applicability}</p>
        </div>
        {config.warnings.map((warning) => (
          <div
            key={warning}
            className="flex items-start gap-2 rounded-md border border-amber-500/30 bg-amber-500/5 p-3 text-xs"
            role="note"
          >
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
            <span>{warning}</span>
          </div>
        ))}
        {config.fields.map((field) => {
          const fieldId = `${config.id}-${field.id}`;

          return (
            <div key={field.id}>
              <label className="block text-sm font-medium" htmlFor={fieldId}>
                {field.label}
              </label>
              {field.type === "select" ? (
                <Select
                  id={fieldId}
                  className="mt-1"
                  value={values[field.id] as string}
                  options={field.options ?? []}
                  onValueChange={(value) => setValue(field.id, value)}
                />
              ) : field.type === "checkbox" ? (
                <label className="mt-2 flex h-6 w-fit cursor-pointer items-center gap-2.5 text-sm leading-none text-foreground">
                  <Checkbox
                    id={fieldId}
                    checked={values[field.id] as boolean}
                    onChange={(e) => setValue(field.id, e.target.checked)}
                  />
                  Да
                </label>
              ) : (
                <Input
                  id={fieldId}
                  type="number"
                  className="mt-1"
                  min={field.min}
                  max={field.max}
                  step="any"
                  value={String(values[field.id] ?? "")}
                  onChange={(e) => setValue(field.id, e.target.value)}
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
                {result.value} {result.unit}
              </p>
              <p className="text-sm text-muted-foreground">
                {result.interpretation}
              </p>
              {result.details.length > 0 && (
                <ul className="space-y-0.5 text-xs text-muted-foreground">
                  {result.details.map((detail) => <li key={detail}>{detail}</li>)}
                </ul>
              )}
              {result.warnings.map((warning) => (
                <p key={warning} className="text-xs text-amber-700 dark:text-amber-400">
                  {warning}
                </p>
              ))}
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
        <div className="flex items-start gap-2 border-t pt-3 text-xs text-muted-foreground">
          <BookOpen className="mt-0.5 h-3.5 w-3.5 shrink-0" />
          <span>
            Источник: {config.reference}. Версия: {config.version}.
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
