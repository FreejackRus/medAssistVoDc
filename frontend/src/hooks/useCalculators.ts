import { useMutation } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";

export interface CalculatorResult {
  result: number;
  unit: string;
  interpretation: string;
}

export interface CalculatorConfig {
  id: string;
  groupId: CalculatorGroupId;
  title: string;
  description: string;
  endpoint: string;
  fields: FieldConfig[];
}

export type CalculatorGroupId =
  | "anthropometry"
  | "renal-function"
  | "cardiovascular-risk"
  | "medication-dosing";

export interface CalculatorGroup {
  id: CalculatorGroupId;
  title: string;
  description: string;
}

export interface FieldConfig {
  name: string;
  label: string;
  type: "number" | "select" | "checkbox";
  placeholder?: string;
  options?: { value: string; label: string }[];
  defaultValue?: string | number | boolean;
  min?: number;
  max?: number;
}

export const calculatorGroups: CalculatorGroup[] = [
  {
    id: "anthropometry",
    title: "Антропометрия",
    description: "Показатели роста, массы и площади поверхности тела",
  },
  {
    id: "renal-function",
    title: "Функция почек",
    description: "Расчётные показатели функции почек",
  },
  {
    id: "cardiovascular-risk",
    title: "Сердечно-сосудистый риск",
    description: "Оценка риска сердечно-сосудистых событий",
  },
  {
    id: "medication-dosing",
    title: "Дозирование препаратов",
    description: "Расчёт дозы с учётом параметров пациента",
  },
];

export const calculators: CalculatorConfig[] = [
  {
    id: "bmi",
    groupId: "anthropometry",
    title: "Индекс массы тела (ИМТ)",
    description: "Оценка массы тела по формуле Кетле",
    endpoint: "/calculators/bmi",
    fields: [
      { name: "height_cm", label: "Рост (см)", type: "number", placeholder: "170", defaultValue: 170, min: 50, max: 300 },
      { name: "weight_kg", label: "Вес (кг)", type: "number", placeholder: "70", defaultValue: 70, min: 1, max: 500 },
    ],
  },
  {
    id: "creatinine",
    groupId: "renal-function",
    title: "Клиренс креатинина",
    description: "Формула Кокрофта-Голта",
    endpoint: "/calculators/creatinine",
    fields: [
      { name: "age", label: "Возраст (лет)", type: "number", placeholder: "50", defaultValue: 50, min: 1, max: 150 },
      { name: "weight_kg", label: "Вес (кг)", type: "number", placeholder: "70", defaultValue: 70, min: 1, max: 500 },
      { name: "creatinine", label: "Креатинин (мкмоль/л)", type: "number", placeholder: "80", defaultValue: 80, min: 1, max: 2000 },
      {
        name: "gender",
        label: "Пол",
        type: "select",
        options: [
          { value: "male", label: "Мужской" },
          { value: "female", label: "Женский" },
        ],
        defaultValue: "male",
      },
    ],
  },
  {
    id: "bsa",
    groupId: "anthropometry",
    title: "Площадь поверхности тела",
    description: "Формула Дю Буа",
    endpoint: "/calculators/bsa",
    fields: [
      { name: "height_cm", label: "Рост (см)", type: "number", placeholder: "170", defaultValue: 170, min: 50, max: 300 },
      { name: "weight_kg", label: "Вес (кг)", type: "number", placeholder: "70", defaultValue: 70, min: 1, max: 500 },
    ],
  },
  {
    id: "dosage",
    groupId: "medication-dosing",
    title: "Расчёт дозировки",
    description: "Расчёт дозы препарата по массе тела",
    endpoint: "/calculators/dosage",
    fields: [
      { name: "weight_kg", label: "Вес (кг)", type: "number", placeholder: "70", defaultValue: 70, min: 1, max: 500 },
      { name: "dose_per_kg", label: "Доза на кг (мг/кг)", type: "number", placeholder: "5", defaultValue: 5, min: 0.01, max: 1000 },
      { name: "frequency", label: "Кратность приёма (раз/сут)", type: "number", placeholder: "2", defaultValue: 2, min: 1, max: 24 },
    ],
  },
  {
    id: "score",
    groupId: "cardiovascular-risk",
    title: "SCORE (учебная версия)",
    description: "Упрощённая учебная оценка 10-летнего сердечно-сосудистого риска. Не использовать для клинических решений.",
    endpoint: "/calculators/score",
    fields: [
      { name: "age", label: "Возраст (лет)", type: "number", placeholder: "55", defaultValue: 55, min: 40, max: 65 },
      {
        name: "gender",
        label: "Пол",
        type: "select",
        options: [
          { value: "male", label: "Мужской" },
          { value: "female", label: "Женский" },
        ],
        defaultValue: "male",
      },
      { name: "smoking", label: "Курение", type: "checkbox", defaultValue: false },
      { name: "cholesterol", label: "Общий холестерин (ммоль/л)", type: "number", placeholder: "5.5", defaultValue: 5.5, min: 2, max: 20 },
      { name: "systolic_bp", label: "Систолическое АД (мм рт. ст.)", type: "number", placeholder: "140", defaultValue: 140, min: 80, max: 300 },
    ],
  },
];

export function useCalculate(endpoint: string) {
  return useMutation<CalculatorResult, Error, Record<string, unknown>>({
    mutationFn: (body) =>
      apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify(body),
      }),
  });
}
