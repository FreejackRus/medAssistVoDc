use axum::{Json, extract::Path};
use chrono::{Duration, NaiveDate};
use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::{auth::AuthUser, error::AppError};

const REGISTRY_VERSION: &str = "2026.1";

#[derive(Clone, Serialize)]
pub struct SelectOption {
    value: String,
    label: String,
}

#[derive(Clone, Serialize)]
pub struct CalculatorField {
    id: String,
    #[serde(rename = "type")]
    kind: String,
    label: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<Vec<SelectOption>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    default: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct CalculatorMetadata {
    id: String,
    group: String,
    title: String,
    description: String,
    fields: Vec<CalculatorField>,
    warnings: Vec<String>,
    applicability: String,
    reference: String,
    version: String,
}

#[derive(Serialize)]
pub struct CalculatorGroup {
    id: String,
    title: String,
    calculators: Vec<CalculatorMetadata>,
}

#[derive(Serialize)]
pub struct CalculatorRegistry {
    version: String,
    count: usize,
    groups: Vec<CalculatorGroup>,
}

#[derive(Debug, Serialize)]
pub struct CalculatorResult {
    value: Value,
    unit: String,
    interpretation: String,
    details: Vec<String>,
    warnings: Vec<String>,
    reference: String,
}

fn ru(text: &str) -> String {
    match text {
        "Sex" => "Пол",
        "Male" => "Мужской",
        "Female" => "Женский",
        "Body mass index" => "Индекс массы тела (ИМТ/BMI)",
        "Weight adjusted for height squared." => "Масса тела с поправкой на квадрат роста.",
        "Height (cm)" => "Рост (см)",
        "Weight (kg)" => "Масса тела (кг)",
        "Adults; interpretation is not a direct measure of body composition." => "Взрослые; показатель не является прямой оценкой состава тела.",
        "Quetelet body mass index formula." => "Формула индекса массы тела Кетле.",
        "Body surface area (Du Bois)" => "Площадь поверхности тела (Дюбуа)",
        "Estimated body surface area from height and weight." => "Расчётная площадь поверхности тела по росту и массе.",
        "Validated historically in a small adult sample; use caution at body-size extremes." => "Исторически валидировано на небольшой выборке взрослых; при крайних размерах тела требуется осторожность.",
        "Ideal body weight (Devine)" => "Идеальная масса тела (Devine)",
        "Height-based ideal body weight estimate." => "Расчёт идеальной массы тела по росту.",
        "Adults; not validated for children or as a nutritional target." => "Взрослые; не валидировано для детей или как целевой показатель питания.",
        "Prescribed mg/kg amount" => "Расчёт назначенного количества в мг/кг",
        "Arithmetic conversion of an already prescribed mg/kg amount." => "Арифметический пересчёт уже назначенного количества в мг/кг.",
        "Prescribed amount (mg/kg)" => "Назначенное количество (мг/кг)",
        "This tool does not recommend a drug, dose, route, interval, or maximum." => "Инструмент не рекомендует препарат, дозу, путь или интервал введения либо максимальную дозу.",
        "Use only to calculate an amount from an independently verified prescription." => "Использовать только для пересчёта независимо проверенного назначения.",
        "Dimensional calculation: kg × mg/kg." => "Размерностный расчёт: кг × мг/кг.",
        "Mean arterial pressure" => "Среднее артериальное давление",
        "Resting-cycle approximation from systolic and diastolic pressure." => "Приближённый расчёт по систолическому и диастолическому давлению.",
        "Systolic BP (mmHg)" => "Систолическое АД (мм рт. ст.)",
        "Diastolic BP (mmHg)" => "Диастолическое АД (мм рт. ст.)",
        "Approximation is most reliable at normal heart rates." => "Приближение наиболее надёжно при нормальной частоте сердечных сокращений.",
        "MAP ≈ (SBP + 2 × DBP) / 3." => "САДср ≈ (САД + 2 × ДАД) / 3.",
        "Stroke-risk factor score for atrial fibrillation." => "Шкала факторов риска инсульта при фибрилляции предсердий.",
        "Age (years)" => "Возраст (лет)",
        "Heart failure/LV dysfunction" => "Сердечная недостаточность/дисфункция ЛЖ",
        "Hypertension" => "Артериальная гипертензия",
        "Diabetes mellitus" => "Сахарный диабет",
        "Prior stroke/TIA/systemic embolism" => "Инсульт/ТИА/системная эмболия в анамнезе",
        "Vascular disease" => "Сосудистое заболевание",
        "Adults with non-valvular atrial fibrillation; treatment decisions require current guidance." => "Взрослые с неклапанной фибрилляцией предсердий; решения о лечении принимают по актуальным рекомендациям.",
        "One-year major bleeding risk-factor score." => "Шкала факторов риска большого кровотечения в течение года.",
        "Uncontrolled hypertension" => "Неконтролируемая артериальная гипертензия",
        "Abnormal renal function" => "Нарушение функции почек",
        "Abnormal liver function" => "Нарушение функции печени",
        "Prior stroke" => "Инсульт в анамнезе",
        "Bleeding history/predisposition" => "Кровотечение в анамнезе/предрасположенность",
        "Labile INR" => "Лабильное МНО",
        "Age over 65" => "Возраст старше 65 лет",
        "Antiplatelet/NSAID use" => "Приём антиагрегантов/НПВП",
        "Alcohol use" => "Употребление алкоголя",
        "A high score identifies modifiable risk and closer review; it is not by itself a reason to withhold anticoagulation." => "Высокий балл указывает на модифицируемые риски и необходимость более тщательного наблюдения; сам по себе он не является основанием для отказа от антикоагуляции.",
        "Adults with atrial fibrillation being considered for anticoagulation." => "Взрослые с фибрилляцией предсердий, у которых рассматривается антикоагуляция.",
        "Corrected QT interval" => "Корригированный интервал QT (QTc)",
        "Heart-rate corrected QT by selectable formula." => "Коррекция QT по частоте сердечных сокращений выбранной формулой.",
        "QT interval (ms)" => "Интервал QT (мс)",
        "Heart rate (beats/min)" => "ЧСС (уд/мин)",
        "Correction formula" => "Формула коррекции",
        "Bazett" => "Базетт",
        "Fridericia" => "Фридеричиа",
        "Bazett correction can overcorrect at high and undercorrect at low heart rates." => "Формула Базетта может завышать QTc при высокой и занижать при низкой ЧСС.",
        "Measured QT and regular rhythm; automated measurements require verification." => "Измеренный QT при регулярном ритме; автоматические измерения требуют проверки.",
        "Creatinine clearance (Cockcroft–Gault)" => "Клиренс креатинина (Кокрофт—Голт)",
        "Creatinine clearance estimate using age, weight, sex, and serum creatinine." => "Расчёт клиренса креатинина по возрасту, массе, полу и креатинину сыворотки.",
        "Weight used in equation (kg)" => "Масса для расчёта (кг)",
        "Serum creatinine (mg/dL)" => "Креатинин сыворотки (мг/дл)",
        "Choice of actual, ideal, or adjusted weight depends on clinical context." => "Выбор фактической, идеальной или скорректированной массы зависит от клинической ситуации.",
        "Adults with stable kidney function; not indexed to body surface area." => "Взрослые со стабильной функцией почек; результат не индексирован по площади поверхности тела.",
        "CKD-EPI 2021 eGFR" => "рСКФ CKD-EPI 2021",
        "Race-free creatinine-based eGFR estimate." => "Расчёт рСКФ по креатинину без расового коэффициента.",
        "Adults with stable creatinine; not for acute kidney injury, pregnancy, or children." => "Взрослые со стабильным креатинином; не применять при ОПП, беременности и у детей.",
        "Fractional excretion of sodium" => "Фракционная экскреция натрия (FENa)",
        "Fraction of filtered sodium excreted in urine." => "Доля профильтрованного натрия, выведенная с мочой.",
        "Urine sodium (mmol/L)" => "Натрий мочи (ммоль/л)",
        "Plasma sodium (mmol/L)" => "Натрий плазмы (ммоль/л)",
        "Urine creatinine" => "Креатинин мочи",
        "Plasma creatinine (same unit)" => "Креатинин плазмы (в тех же единицах)",
        "Diuretics and several renal disorders can limit interpretation." => "Диуретики и некоторые заболевания почек могут ограничивать интерпретацию.",
        "Paired blood and urine samples; creatinine values must use the same units." => "Парные пробы крови и мочи; значения креатинина должны быть в одинаковых единицах.",
        "Fractional excretion of urea" => "Фракционная экскреция мочевины (FEUrea)",
        "Fraction of filtered urea excreted in urine." => "Доля профильтрованной мочевины, выведенная с мочой.",
        "Urine urea" => "Мочевина мочи",
        "Plasma urea (same unit)" => "Мочевина плазмы (в тех же единицах)",
        "Sepsis and other conditions can limit interpretation." => "Сепсис и другие состояния могут ограничивать интерпретацию.",
        "Paired blood and urine samples; each analyte pair must use matching units." => "Парные пробы крови и мочи; единицы в каждой паре аналитов должны совпадать.",
        "Pneumonia severity risk-factor count." => "Подсчёт факторов тяжести пневмонии.",
        "New confusion" => "Впервые возникшая спутанность сознания",
        "Urea (mmol/L)" => "Мочевина (ммоль/л)",
        "Respiratory rate (/min)" => "Частота дыхания (/мин)",
        "Adults with community-acquired pneumonia; does not replace clinical judgment." => "Взрослые с внебольничной пневмонией; не заменяет клиническое суждение.",
        "P/F ratio" => "Отношение P/F",
        "Arterial oxygen tension divided by inspired oxygen fraction." => "Отношение напряжения кислорода в артериальной крови к доле кислорода во вдыхаемой смеси.",
        "ARDS severity categories additionally require clinical and ventilatory criteria." => "Категории тяжести ОРДС дополнительно требуют клинических и вентиляционных критериев.",
        "Arterial blood gas paired with the current FiO₂." => "Газовый состав артериальной крови при текущей FiO₂.",
        "Alveolar–arterial oxygen gradient" => "Альвеолярно-артериальный градиент кислорода",
        "Sea-level alveolar gas equation estimate." => "Расчёт по уравнению альвеолярного газа.",
        "Atmospheric pressure (mmHg)" => "Атмосферное давление (мм рт. ст.)",
        "PaO₂ (mmHg)" => "PaO₂ (мм рт. ст.)",
        "PaCO₂ (mmHg)" => "PaCO₂ (мм рт. ст.)",
        "FiO₂ (%)" => "Доля кислорода FiO₂ (%)",
        "Uses respiratory quotient 0.8 and water-vapor pressure 47 mmHg." => "Используются дыхательный коэффициент 0,8 и давление водяного пара 47 мм рт. ст.",
        "Blood gas at known FiO₂ and ambient pressure." => "Газовый состав крови при известных FiO₂ и атмосферном давлении.",
        "Smoking pack-years" => "Индекс пачка-лет",
        "Average packs per day multiplied by years smoked." => "Среднее число пачек в день, умноженное на число лет курения.",
        "Packs per day" => "Пачек в день",
        "Years smoked" => "Стаж курения (лет)",
        "Cumulative cigarette exposure; one pack is conventionally 20 cigarettes." => "Суммарная табачная нагрузка; условно одна пачка содержит 20 сигарет.",
        "Pack-years = packs/day × years." => "Пачка-лет = пачек/день × годы.",
        "Glasgow Coma Scale" => "Шкала комы Глазго (GCS)",
        "Eye, verbal, and motor response score." => "Оценка открывания глаз, речевой и двигательной реакции.",
        "Eye response" => "Открывание глаз",
        "Verbal response" => "Речевая реакция",
        "Motor response" => "Двигательная реакция",
        "None (1)" => "Отсутствует (1)",
        "To pressure (2)" => "На боль (2)",
        "To sound (3)" => "На обращение (3)",
        "Spontaneous (4)" => "Спонтанно (4)",
        "Sounds (2)" => "Нечленораздельные звуки (2)",
        "Words (3)" => "Отдельные слова (3)",
        "Confused (4)" => "Спутанная речь (4)",
        "Oriented (5)" => "Ориентирован (5)",
        "Extension (2)" => "Разгибание (2)",
        "Abnormal flexion (3)" => "Патологическое сгибание (3)",
        "Normal flexion (4)" => "Нормальное сгибание (4)",
        "Localizes (5)" => "Локализует боль (5)",
        "Obeys commands (6)" => "Выполняет команды (6)",
        "Document component scores and factors that prevent assessment." => "Документируйте компоненты шкалы и факторы, мешающие оценке.",
        "Consciousness assessment after accounting for sedation, paralysis, and intubation." => "Оценка сознания с учётом седации, миорелаксации и интубации.",
        "Three bedside risk criteria in suspected infection." => "Три прикроватных критерия риска при подозрении на инфекцию.",
        "Altered mentation" => "Изменение сознания",
        "qSOFA is not a stand-alone sepsis screening or diagnostic test." => "qSOFA не является самостоятельным скрининговым или диагностическим тестом на сепсис.",
        "Adults with suspected infection outside the ICU." => "Взрослые с подозрением на инфекцию вне ОРИТ.",
        "SIRS criteria" => "Критерии SIRS",
        "Count of systemic inflammatory response criteria." => "Подсчёт критериев синдрома системной воспалительной реакции.",
        "Temperature (°C)" => "Температура (°C)",
        "Heart rate (/min)" => "ЧСС (/мин)",
        "WBC (×10⁹/L)" => "Лейкоциты (×10⁹/л)",
        "Immature bands (%)" => "Незрелые палочкоядерные формы (%)",
        "SIRS is nonspecific and is not the current definition of sepsis." => "SIRS неспецифичен и не является современным определением сепсиса.",
        "Patients evaluated for systemic inflammation." => "Пациенты, обследуемые по поводу системного воспаления.",
        "Shock index" => "Шоковый индекс",
        "Heart rate divided by systolic blood pressure." => "Отношение ЧСС к систолическому АД.",
        "Adjunctive hemodynamic assessment; thresholds vary by population." => "Дополнительная оценка гемодинамики; пороги зависят от популяции.",
        "Child–Pugh score" => "Шкала Чайлд—Пью",
        "Five-component cirrhosis severity score." => "Пятикомпонентная шкала тяжести цирроза.",
        "Total bilirubin (mg/dL)" => "Общий билирубин (мг/дл)",
        "Albumin (g/dL)" => "Альбумин (г/дл)",
        "INR" => "МНО",
        "Ascites" => "Асцит",
        "Encephalopathy" => "Энцефалопатия",
        "None" => "Нет",
        "Mild/controlled" => "Лёгкий/контролируемый",
        "Moderate-severe/refractory" => "Умеренный-тяжёлый/рефрактерный",
        "Grade I–II" => "Степень I–II",
        "Grade III–IV" => "Степень III–IV",
        "Clinical component grading can be subjective." => "Оценка клинических компонентов может быть субъективной.",
        "Cirrhosis; bilirubin thresholds shown are not disease-specific variants." => "Цирроз; приведённые пороги билирубина не учитывают варианты для отдельных заболеваний.",
        "Current sex-, albumin-, sodium-, INR-, creatinine-, and bilirubin-based model." => "Современная модель по полу, альбумину, натрию, МНО, креатинину и билирубину.",
        "Creatinine (mg/dL)" => "Креатинин (мг/дл)",
        "Sodium (mmol/L)" => "Натрий (ммоль/л)",
        "Dialysis at least twice in prior week" => "Диализ не менее двух раз за предыдущую неделю",
        "Allocation systems may apply additional policy rules; verify the official system used locally." => "Системы распределения могут применять дополнительные правила; сверяйтесь с официальной локальной системой.",
        "Candidates with chronic liver disease; laboratory values use official equation caps." => "Пациенты с хроническим заболеванием печени; лабораторные значения ограничиваются официальными пределами формулы.",
        "Age and routine laboratory estimate of advanced fibrosis risk." => "Оценка риска выраженного фиброза по возрасту и рутинным анализам.",
        "AST (U/L)" => "АСТ (Ед/л)",
        "ALT (U/L)" => "АЛТ (Ед/л)",
        "Platelets (×10⁹/L)" => "Тромбоциты (×10⁹/л)",
        "Interpretive cutoffs depend on age, disease, and care pathway." => "Пороги интерпретации зависят от возраста, заболевания и клинического маршрута.",
        "Adults with chronic liver disease; less reliable in acute hepatitis." => "Взрослые с хроническим заболеванием печени; менее надёжно при остром гепатите.",
        "Fasting glucose-insulin homeostasis estimate." => "Оценка глюкозо-инсулинового гомеостаза натощак.",
        "Fasting glucose (mmol/L)" => "Глюкоза натощак (ммоль/л)",
        "Fasting insulin (mIU/L)" => "Инсулин натощак (мМЕ/л)",
        "Assay- and population-specific thresholds vary." => "Пороги зависят от метода анализа и популяции.",
        "Fasting, metabolically stable patients; not a diagnostic test by itself." => "Метаболически стабильные пациенты натощак; отдельно не является диагностическим тестом.",
        "Estimated average glucose" => "Расчётная средняя глюкоза",
        "HbA1c-derived average glucose estimate." => "Расчёт средней глюкозы по HbA1c.",
        "HbA1c (%)" => "Гликированный гемоглобин HbA1c (%)",
        "May be inaccurate when red-cell turnover or hemoglobin variants affect HbA1c." => "Может быть неточным при изменённом обмене эритроцитов или вариантах гемоглобина, влияющих на HbA1c.",
        "Use when HbA1c reliably reflects glycemia." => "Применять, когда HbA1c надёжно отражает гликемию.",
        "Corrected sodium in hyperglycemia" => "Скорректированный натрий при гипергликемии",
        "Sodium corrected by 1.6 mmol/L per 100 mg/dL glucose above 100." => "Коррекция натрия на 1,6 ммоль/л на каждые 100 мг/дл глюкозы свыше 100.",
        "Measured sodium (mmol/L)" => "Измеренный натрий (ммоль/л)",
        "Glucose (mg/dL)" => "Глюкоза (мг/дл)",
        "The correction factor is an approximation and differs in severe hyperglycemia." => "Коэффициент коррекции приблизителен и может отличаться при тяжёлой гипергликемии.",
        "Hyperglycemia with glucose reported in mg/dL." => "Гипергликемия при концентрации глюкозы в мг/дл.",
        "Absolute neutrophil count" => "Абсолютное число нейтрофилов (ANC)",
        "WBC multiplied by segmented neutrophil and band fraction." => "Число лейкоцитов, умноженное на долю сегментоядерных и палочкоядерных нейтрофилов.",
        "Neutrophils (%)" => "Нейтрофилы (%)",
        "Bands (%)" => "Палочкоядерные нейтрофилы (%)",
        "CBC differential; entered percentages must sum to no more than 100." => "Лейкоцитарная формула; сумма введённых процентов не должна превышать 100.",
        "Mentzer index" => "Индекс Ментцера",
        "MCV divided by red blood cell count." => "Отношение MCV к числу эритроцитов.",
        "MCV (fL)" => "MCV (фл)",
        "RBC (×10¹²/L)" => "Эритроциты (×10¹²/л)",
        "Screening heuristic only; iron studies and hemoglobin analysis establish etiology." => "Только скрининговый ориентир; этиологию устанавливают по обмену железа и анализу гемоглобина.",
        "Microcytic anemia." => "Микроцитарная анемия.",
        "Wells score for pulmonary embolism" => "Шкала Wells для ТЭЛА",
        "Clinical pretest-probability score for pulmonary embolism." => "Шкала клинической предтестовой вероятности ТЭЛА.",
        "Clinical signs of DVT" => "Клинические признаки ТГВ",
        "PE more likely than alternative diagnosis" => "ТЭЛА вероятнее альтернативного диагноза",
        "Immobilization ≥3 days or surgery in prior 4 weeks" => "Иммобилизация ≥3 дней или операция в предыдущие 4 недели",
        "Prior DVT/PE" => "ТГВ/ТЭЛА в анамнезе",
        "Hemoptysis" => "Кровохарканье",
        "Active/recently treated malignancy" => "Активная или недавно леченная злокачественная опухоль",
        "Use the interpretation pathway validated by the local diagnostic protocol." => "Использовать схему интерпретации, валидированную локальным диагностическим протоколом.",
        "Hemodynamically stable adults with suspected pulmonary embolism." => "Гемодинамически стабильные взрослые с подозрением на ТЭЛА.",
        "Estimated due date (Naegele)" => "Предполагаемая дата родов (Негеле)",
        "Adds 280 days to the first day of the last menstrual period." => "Добавляет 280 дней к первому дню последней менструации.",
        "LMP year" => "Год последней менструации",
        "LMP month" => "Месяц последней менструации",
        "LMP day" => "День последней менструации",
        "Cycle length and ultrasound dating can change the best obstetric estimate." => "Длина цикла и датировка по УЗИ могут изменить оптимальную акушерскую оценку.",
        "Regular 28-day cycle with known first day of last menstrual period." => "Регулярный 28-дневный цикл и известный первый день последней менструации.",
        "Naegele rule: LMP + 280 days." => "Правило Негеле: ПМ + 280 дней.",
        "Bishop score" => "Шкала Бишопа",
        "Cervical examination score before induction." => "Оценка шейки матки перед индукцией родов.",
        "Dilation (cm)" => "Раскрытие (см)",
        "Effacement (%)" => "Сглаженность (%)",
        "Fetal station" => "Положение предлежащей части",
        "−1 or 0" => "−1 или 0",
        "+1 or +2" => "+1 или +2",
        "Cervical consistency" => "Консистенция шейки матки",
        "Firm" => "Плотная",
        "Medium" => "Средняя",
        "Soft" => "Мягкая",
        "Cervical position" => "Положение шейки матки",
        "Posterior" => "Кзади",
        "Mid-position" => "Срединное",
        "Anterior" => "Кпереди",
        "Exam findings are subjective and management depends on the full obstetric context." => "Данные осмотра субъективны; тактика зависит от полной акушерской ситуации.",
        "Term obstetric patients being assessed for induction." => "Доношенная беременность при оценке перед индукцией.",
        "Bedside Schwartz eGFR" => "Прикроватная рСКФ Schwartz",
        "Height and creatinine pediatric eGFR estimate." => "Расчёт рСКФ у детей по росту и креатинину.",
        "Children and adolescents with stable creatinine measured using an IDMS-traceable assay." => "Дети и подростки со стабильным креатинином, измеренным методом, прослеживаемым к IDMS.",
        "Holliday–Segar maintenance fluids" => "Поддерживающая инфузия Holliday–Segar",
        "Weight-based daily maintenance water estimate." => "Расчёт суточной поддерживающей потребности в воде по массе.",
        "Maintenance estimates do not include deficits, ongoing losses, resuscitation, or disease-specific restriction." => "Расчёт не учитывает дефицит, продолжающиеся потери, реанимацию и ограничения при отдельных заболеваниях.",
        "Children without conditions requiring a modified fluid strategy." => "Дети без состояний, требующих изменения инфузионной стратегии.",
        "General practice" => "Общая практика",
        "Cardiology" => "Кардиология",
        "Nephrology" => "Нефрология",
        "Pulmonology" => "Пульмонология",
        "Emergency & ICU" => "Неотложная помощь и ОРИТ",
        "Hepatology" => "Гепатология",
        "Endocrinology" => "Эндокринология",
        "Hematology" => "Гематология",
        "Obstetrics" => "Акушерство",
        "Pediatrics" => "Педиатрия",
        "Inker LA et al. 2021 CKD-EPI creatinine equation." => {
            "Inker LA et al. Креатининовое уравнение CKD-EPI 2021."
        }
        "Teasdale G, Jennett B. Glasgow Coma Scale." => {
            "Teasdale G, Jennett B. Шкала комы Глазго."
        }
        "Sepsis-3 qSOFA criteria." => "Критерии qSOFA консенсуса Sepsis-3.",
        "ACCP/SCCM 1992 SIRS criteria." => "Критерии SIRS ACCP/SCCM 1992 года.",
        "Matthews DR et al. HOMA model." => "Matthews DR et al. Модель HOMA.",
        "Nathan DM et al. ADAG relationship." => "Nathan DM et al. Зависимость ADAG.",
        "Wells PS et al. PE clinical model." => "Wells PS et al. Клиническая модель ТЭЛА.",
        "Schwartz GJ et al. 2009 bedside equation." => {
            "Schwartz GJ et al. Прикроватное уравнение 2009 года."
        }
        "Lip GYH et al. CHA₂DS₂-VASc risk-factor scheme." => {
            "Lip GYH et al. Схема факторов риска CHA₂DS₂-VASc."
        }
        "Pisters R et al. HAS-BLED score." => "Pisters R et al. Шкала HAS-BLED.",
        "Cockcroft–Gault equation." => "Уравнение Кокрофта—Голта.",
        "2021 CKD-EPI creatinine equation." => "Креатининовое уравнение CKD-EPI 2021.",
        "FENa formula." => "Формула FENa.",
        "FEUrea formula." => "Формула FEUrea.",
        "Alveolar gas equation." => "Уравнение альвеолярного газа.",
        "MAP approximation." => "Приближённая формула среднего артериального давления.",
        "CHA₂DS₂-VASc risk-factor scheme." => "Схема факторов риска CHA₂DS₂-VASc.",
        "HAS-BLED score." => "Шкала HAS-BLED.",
        "Bazett or Fridericia correction." => "Коррекция по Базетту или Фридеричиа.",
        "Packs/day × years." => "Пачек/день × годы.",
        "Glasgow Coma Scale." => "Шкала комы Глазго.",
        "Sepsis-3 qSOFA." => "Критерии qSOFA консенсуса Sepsis-3.",
        "ACCP/SCCM SIRS criteria." => "Критерии SIRS ACCP/SCCM.",
        "Allgöwer shock index." => "Шоковый индекс Алльговера.",
        "Child–Pugh score." => "Шкала Чайлд—Пью.",
        "FIB-4 formula." => "Формула FIB-4.",
        "HOMA-IR formula." => "Формула HOMA-IR.",
        "ADAG relationship." => "Зависимость ADAG.",
        "Katz correction factor." => "Коэффициент коррекции Katz.",
        "ANC formula." => "Формула ANC.",
        "Mentzer index." => "Индекс Ментцера.",
        "Wells PE score." => "Шкала Wells для ТЭЛА.",
        "Naegele rule." => "Правило Негеле.",
        "Bishop score." => "Шкала Бишопа.",
        "2009 bedside Schwartz equation." => "Прикроватное уравнение Schwartz 2009 года.",
        "Holliday–Segar method." => "Метод Holliday–Segar.",
        _ => text,
    }
    .into()
}

fn number(id: &str, label: &str, min: f64, max: f64, default: Option<f64>) -> CalculatorField {
    CalculatorField {
        id: id.into(),
        kind: "number".into(),
        label: ru(label),
        options: None,
        default: default.map(Value::from),
        min: Some(min),
        max: Some(max),
    }
}

fn checkbox(id: &str, label: &str) -> CalculatorField {
    CalculatorField {
        id: id.into(),
        kind: "checkbox".into(),
        label: ru(label),
        options: None,
        default: Some(Value::Bool(false)),
        min: None,
        max: None,
    }
}

fn select(id: &str, label: &str, options: &[(&str, &str)], default: &str) -> CalculatorField {
    CalculatorField {
        id: id.into(),
        kind: "select".into(),
        label: ru(label),
        options: Some(
            options
                .iter()
                .map(|(value, label)| SelectOption {
                    value: (*value).into(),
                    label: ru(label),
                })
                .collect(),
        ),
        default: Some(Value::String(default.into())),
        min: None,
        max: None,
    }
}

fn metadata(
    id: &str,
    group: &str,
    title: &str,
    description: &str,
    fields: Vec<CalculatorField>,
    warnings: &[&str],
    applicability: &str,
    reference: &str,
) -> CalculatorMetadata {
    CalculatorMetadata {
        id: id.into(),
        group: group.into(),
        title: ru(title),
        description: ru(description),
        fields,
        warnings: warnings.iter().map(|v| ru(v)).collect(),
        applicability: ru(applicability),
        reference: ru(reference),
        version: REGISTRY_VERSION.into(),
    }
}

fn calculators() -> Vec<CalculatorMetadata> {
    let sex = || {
        select(
            "sex",
            "Sex",
            &[("male", "Male"), ("female", "Female")],
            "male",
        )
    };
    vec![
        metadata(
            "bmi",
            "general-practice",
            "Body mass index",
            "Weight adjusted for height squared.",
            vec![
                number("height_cm", "Height (cm)", 50.0, 300.0, None),
                number("weight_kg", "Weight (kg)", 1.0, 500.0, None),
            ],
            &[],
            "Adults; interpretation is not a direct measure of body composition.",
            "Quetelet body mass index formula.",
        ),
        metadata(
            "bsa-du-bois",
            "general-practice",
            "Body surface area (Du Bois)",
            "Estimated body surface area from height and weight.",
            vec![
                number("height_cm", "Height (cm)", 50.0, 300.0, None),
                number("weight_kg", "Weight (kg)", 1.0, 500.0, None),
            ],
            &[],
            "Validated historically in a small adult sample; use caution at body-size extremes.",
            "Du Bois D, Du Bois EF (1916).",
        ),
        metadata(
            "ideal-body-weight-devine",
            "general-practice",
            "Ideal body weight (Devine)",
            "Height-based ideal body weight estimate.",
            vec![
                sex(),
                number("height_cm", "Height (cm)", 120.0, 250.0, None),
            ],
            &[],
            "Adults; not validated for children or as a nutritional target.",
            "Devine BJ (1974).",
        ),
        metadata(
            "mg-kg-dose",
            "general-practice",
            "Prescribed mg/kg amount",
            "Arithmetic conversion of an already prescribed mg/kg amount.",
            vec![
                number("weight_kg", "Weight (kg)", 0.1, 500.0, None),
                number("dose_mg_kg", "Prescribed amount (mg/kg)", 0.0, 1000.0, None),
            ],
            &["This tool does not recommend a drug, dose, route, interval, or maximum."],
            "Use only to calculate an amount from an independently verified prescription.",
            "Dimensional calculation: kg × mg/kg.",
        ),
        metadata(
            "mean-arterial-pressure",
            "cardiology",
            "Mean arterial pressure",
            "Resting-cycle approximation from systolic and diastolic pressure.",
            vec![
                number("systolic_bp", "Systolic BP (mmHg)", 30.0, 300.0, None),
                number("diastolic_bp", "Diastolic BP (mmHg)", 10.0, 200.0, None),
            ],
            &[],
            "Approximation is most reliable at normal heart rates.",
            "MAP ≈ (SBP + 2 × DBP) / 3.",
        ),
        metadata(
            "cha2ds2-vasc",
            "cardiology",
            "CHA₂DS₂-VASc",
            "Stroke-risk factor score for atrial fibrillation.",
            vec![
                number("age", "Age (years)", 18.0, 120.0, None),
                sex(),
                checkbox("heart_failure", "Heart failure/LV dysfunction"),
                checkbox("hypertension", "Hypertension"),
                checkbox("diabetes", "Diabetes mellitus"),
                checkbox("stroke_tia", "Prior stroke/TIA/systemic embolism"),
                checkbox("vascular_disease", "Vascular disease"),
            ],
            &[],
            "Adults with non-valvular atrial fibrillation; treatment decisions require current guidance.",
            "Lip GYH et al. CHA₂DS₂-VASc risk-factor scheme.",
        ),
        metadata(
            "has-bled",
            "cardiology",
            "HAS-BLED",
            "One-year major bleeding risk-factor score.",
            vec![
                checkbox("hypertension", "Uncontrolled hypertension"),
                checkbox("abnormal_renal", "Abnormal renal function"),
                checkbox("abnormal_liver", "Abnormal liver function"),
                checkbox("stroke", "Prior stroke"),
                checkbox("bleeding", "Bleeding history/predisposition"),
                checkbox("labile_inr", "Labile INR"),
                checkbox("age_over_65", "Age over 65"),
                checkbox("drugs", "Antiplatelet/NSAID use"),
                checkbox("alcohol", "Alcohol use"),
            ],
            &[
                "A high score identifies modifiable risk and closer review; it is not by itself a reason to withhold anticoagulation.",
            ],
            "Adults with atrial fibrillation being considered for anticoagulation.",
            "Pisters R et al. HAS-BLED score.",
        ),
        metadata(
            "qtc",
            "cardiology",
            "Corrected QT interval",
            "Heart-rate corrected QT by selectable formula.",
            vec![
                number("qt_ms", "QT interval (ms)", 100.0, 1000.0, None),
                number("heart_rate", "Heart rate (beats/min)", 20.0, 250.0, None),
                select(
                    "formula",
                    "Correction formula",
                    &[("bazett", "Bazett"), ("fridericia", "Fridericia")],
                    "fridericia",
                ),
            ],
            &["Bazett correction can overcorrect at high and undercorrect at low heart rates."],
            "Measured QT and regular rhythm; automated measurements require verification.",
            "Bazett HC (1920); Fridericia LS (1920).",
        ),
        metadata(
            "cockcroft-gault",
            "nephrology",
            "Creatinine clearance (Cockcroft–Gault)",
            "Creatinine clearance estimate using age, weight, sex, and serum creatinine.",
            vec![
                number("age", "Age (years)", 18.0, 120.0, None),
                number(
                    "weight_kg",
                    "Weight used in equation (kg)",
                    20.0,
                    500.0,
                    None,
                ),
                number(
                    "creatinine_mg_dl",
                    "Serum creatinine (mg/dL)",
                    0.1,
                    20.0,
                    None,
                ),
                sex(),
            ],
            &["Choice of actual, ideal, or adjusted weight depends on clinical context."],
            "Adults with stable kidney function; not indexed to body surface area.",
            "Cockcroft DW, Gault MH (1976).",
        ),
        metadata(
            "ckd-epi-2021",
            "nephrology",
            "CKD-EPI 2021 eGFR",
            "Race-free creatinine-based eGFR estimate.",
            vec![
                number("age", "Age (years)", 18.0, 120.0, None),
                number(
                    "creatinine_mg_dl",
                    "Serum creatinine (mg/dL)",
                    0.1,
                    20.0,
                    None,
                ),
                sex(),
            ],
            &[],
            "Adults with stable creatinine; not for acute kidney injury, pregnancy, or children.",
            "Inker LA et al. 2021 CKD-EPI creatinine equation.",
        ),
        metadata(
            "fena",
            "nephrology",
            "Fractional excretion of sodium",
            "Fraction of filtered sodium excreted in urine.",
            vec![
                number("urine_sodium", "Urine sodium (mmol/L)", 0.1, 500.0, None),
                number("plasma_sodium", "Plasma sodium (mmol/L)", 80.0, 200.0, None),
                number("urine_creatinine", "Urine creatinine", 0.1, 1000.0, None),
                number(
                    "plasma_creatinine",
                    "Plasma creatinine (same unit)",
                    0.1,
                    1000.0,
                    None,
                ),
            ],
            &["Diuretics and several renal disorders can limit interpretation."],
            "Paired blood and urine samples; creatinine values must use the same units.",
            "FENa = 100 × (UNa × PCr) / (PNa × UCr).",
        ),
        metadata(
            "feurea",
            "nephrology",
            "Fractional excretion of urea",
            "Fraction of filtered urea excreted in urine.",
            vec![
                number("urine_urea", "Urine urea", 0.1, 5000.0, None),
                number("plasma_urea", "Plasma urea (same unit)", 0.1, 500.0, None),
                number("urine_creatinine", "Urine creatinine", 0.1, 1000.0, None),
                number(
                    "plasma_creatinine",
                    "Plasma creatinine (same unit)",
                    0.1,
                    1000.0,
                    None,
                ),
            ],
            &["Sepsis and other conditions can limit interpretation."],
            "Paired blood and urine samples; each analyte pair must use matching units.",
            "FEUrea = 100 × (UUrea × PCr) / (PUrea × UCr).",
        ),
        metadata(
            "curb-65",
            "pulmonology",
            "CURB-65",
            "Pneumonia severity risk-factor count.",
            vec![
                checkbox("confusion", "New confusion"),
                number("urea_mmol_l", "Urea (mmol/L)", 0.1, 100.0, None),
                number(
                    "respiratory_rate",
                    "Respiratory rate (/min)",
                    1.0,
                    80.0,
                    None,
                ),
                number("systolic_bp", "Systolic BP (mmHg)", 30.0, 300.0, None),
                number("diastolic_bp", "Diastolic BP (mmHg)", 10.0, 200.0, None),
                number("age", "Age (years)", 18.0, 120.0, None),
            ],
            &[],
            "Adults with community-acquired pneumonia; does not replace clinical judgment.",
            "Lim WS et al. CURB-65.",
        ),
        metadata(
            "pf-ratio",
            "pulmonology",
            "P/F ratio",
            "Arterial oxygen tension divided by inspired oxygen fraction.",
            vec![
                number("pao2_mm_hg", "PaO₂ (mmHg)", 20.0, 700.0, None),
                number("fio2_percent", "FiO₂ (%)", 21.0, 100.0, None),
            ],
            &["ARDS severity categories additionally require clinical and ventilatory criteria."],
            "Arterial blood gas paired with the current FiO₂.",
            "PaO₂ / FiO₂.",
        ),
        metadata(
            "aa-gradient",
            "pulmonology",
            "Alveolar–arterial oxygen gradient",
            "Sea-level alveolar gas equation estimate.",
            vec![
                number("age", "Age (years)", 0.0, 120.0, None),
                number("fio2_percent", "FiO₂ (%)", 21.0, 100.0, Some(21.0)),
                number("paco2_mm_hg", "PaCO₂ (mmHg)", 5.0, 150.0, None),
                number("pao2_mm_hg", "PaO₂ (mmHg)", 10.0, 700.0, None),
                number(
                    "atmospheric_pressure",
                    "Atmospheric pressure (mmHg)",
                    300.0,
                    800.0,
                    Some(760.0),
                ),
            ],
            &["Uses respiratory quotient 0.8 and water-vapor pressure 47 mmHg."],
            "Blood gas at known FiO₂ and ambient pressure.",
            "Alveolar gas equation.",
        ),
        metadata(
            "smoking-pack-years",
            "pulmonology",
            "Smoking pack-years",
            "Average packs per day multiplied by years smoked.",
            vec![
                number("packs_per_day", "Packs per day", 0.0, 10.0, None),
                number("years_smoked", "Years smoked", 0.0, 100.0, None),
            ],
            &[],
            "Cumulative cigarette exposure; one pack is conventionally 20 cigarettes.",
            "Pack-years = packs/day × years.",
        ),
        metadata(
            "glasgow-coma-scale",
            "emergency-icu",
            "Glasgow Coma Scale",
            "Eye, verbal, and motor response score.",
            vec![
                select(
                    "eye",
                    "Eye response",
                    &[
                        ("1", "None (1)"),
                        ("2", "To pressure (2)"),
                        ("3", "To sound (3)"),
                        ("4", "Spontaneous (4)"),
                    ],
                    "4",
                ),
                select(
                    "verbal",
                    "Verbal response",
                    &[
                        ("1", "None (1)"),
                        ("2", "Sounds (2)"),
                        ("3", "Words (3)"),
                        ("4", "Confused (4)"),
                        ("5", "Oriented (5)"),
                    ],
                    "5",
                ),
                select(
                    "motor",
                    "Motor response",
                    &[
                        ("1", "None (1)"),
                        ("2", "Extension (2)"),
                        ("3", "Abnormal flexion (3)"),
                        ("4", "Normal flexion (4)"),
                        ("5", "Localizes (5)"),
                        ("6", "Obeys commands (6)"),
                    ],
                    "6",
                ),
            ],
            &["Document component scores and factors that prevent assessment."],
            "Consciousness assessment after accounting for sedation, paralysis, and intubation.",
            "Teasdale G, Jennett B. Glasgow Coma Scale.",
        ),
        metadata(
            "qsofa",
            "emergency-icu",
            "qSOFA",
            "Three bedside risk criteria in suspected infection.",
            vec![
                number(
                    "respiratory_rate",
                    "Respiratory rate (/min)",
                    1.0,
                    80.0,
                    None,
                ),
                number("systolic_bp", "Systolic BP (mmHg)", 30.0, 300.0, None),
                checkbox("altered_mentation", "Altered mentation"),
            ],
            &["qSOFA is not a stand-alone sepsis screening or diagnostic test."],
            "Adults with suspected infection outside the ICU.",
            "Sepsis-3 qSOFA criteria.",
        ),
        metadata(
            "sirs",
            "emergency-icu",
            "SIRS criteria",
            "Count of systemic inflammatory response criteria.",
            vec![
                number("temperature_c", "Temperature (°C)", 25.0, 45.0, None),
                number("heart_rate", "Heart rate (/min)", 10.0, 300.0, None),
                number(
                    "respiratory_rate",
                    "Respiratory rate (/min)",
                    1.0,
                    80.0,
                    None,
                ),
                number("paco2_mm_hg", "PaCO₂ (mmHg)", 5.0, 150.0, Some(40.0)),
                number("wbc", "WBC (×10⁹/L)", 0.01, 200.0, None),
                number("bands_percent", "Immature bands (%)", 0.0, 100.0, Some(0.0)),
            ],
            &["SIRS is nonspecific and is not the current definition of sepsis."],
            "Patients evaluated for systemic inflammation.",
            "ACCP/SCCM 1992 SIRS criteria.",
        ),
        metadata(
            "shock-index",
            "emergency-icu",
            "Shock index",
            "Heart rate divided by systolic blood pressure.",
            vec![
                number("heart_rate", "Heart rate (/min)", 10.0, 300.0, None),
                number("systolic_bp", "Systolic BP (mmHg)", 30.0, 300.0, None),
            ],
            &[],
            "Adjunctive hemodynamic assessment; thresholds vary by population.",
            "Allgöwer shock index.",
        ),
        metadata(
            "child-pugh",
            "hepatology",
            "Child–Pugh score",
            "Five-component cirrhosis severity score.",
            vec![
                number(
                    "bilirubin_mg_dl",
                    "Total bilirubin (mg/dL)",
                    0.1,
                    50.0,
                    None,
                ),
                number("albumin_g_dl", "Albumin (g/dL)", 0.5, 6.0, None),
                number("inr", "INR", 0.5, 10.0, None),
                select(
                    "ascites",
                    "Ascites",
                    &[
                        ("none", "None"),
                        ("mild", "Mild/controlled"),
                        ("moderate-severe", "Moderate-severe/refractory"),
                    ],
                    "none",
                ),
                select(
                    "encephalopathy",
                    "Encephalopathy",
                    &[
                        ("none", "None"),
                        ("grade-1-2", "Grade I–II"),
                        ("grade-3-4", "Grade III–IV"),
                    ],
                    "none",
                ),
            ],
            &["Clinical component grading can be subjective."],
            "Cirrhosis; bilirubin thresholds shown are not disease-specific variants.",
            "Child CG; Pugh RNH et al.",
        ),
        metadata(
            "meld-3",
            "hepatology",
            "MELD 3.0",
            "Current sex-, albumin-, sodium-, INR-, creatinine-, and bilirubin-based model.",
            vec![
                number(
                    "bilirubin_mg_dl",
                    "Total bilirubin (mg/dL)",
                    0.1,
                    80.0,
                    None,
                ),
                number("inr", "INR", 0.5, 20.0, None),
                number("creatinine_mg_dl", "Creatinine (mg/dL)", 0.1, 20.0, None),
                number("sodium_mmol_l", "Sodium (mmol/L)", 100.0, 180.0, None),
                number("albumin_g_dl", "Albumin (g/dL)", 0.5, 6.0, None),
                sex(),
                checkbox(
                    "dialysis_twice_weekly",
                    "Dialysis at least twice in prior week",
                ),
            ],
            &[
                "Allocation systems may apply additional policy rules; verify the official system used locally.",
            ],
            "Candidates with chronic liver disease; laboratory values use official equation caps.",
            "Kim WR et al. MELD 3.0 (2021).",
        ),
        metadata(
            "fib-4",
            "hepatology",
            "FIB-4",
            "Age and routine laboratory estimate of advanced fibrosis risk.",
            vec![
                number("age", "Age (years)", 18.0, 120.0, None),
                number("ast_u_l", "AST (U/L)", 1.0, 10000.0, None),
                number("alt_u_l", "ALT (U/L)", 1.0, 10000.0, None),
                number("platelets", "Platelets (×10⁹/L)", 1.0, 2000.0, None),
            ],
            &["Interpretive cutoffs depend on age, disease, and care pathway."],
            "Adults with chronic liver disease; less reliable in acute hepatitis.",
            "Sterling RK et al. FIB-4.",
        ),
        metadata(
            "homa-ir",
            "endocrinology",
            "HOMA-IR",
            "Fasting glucose-insulin homeostasis estimate.",
            vec![
                number(
                    "glucose_mmol_l",
                    "Fasting glucose (mmol/L)",
                    1.0,
                    50.0,
                    None,
                ),
                number(
                    "insulin_miu_l",
                    "Fasting insulin (mIU/L)",
                    0.1,
                    1000.0,
                    None,
                ),
            ],
            &["Assay- and population-specific thresholds vary."],
            "Fasting, metabolically stable patients; not a diagnostic test by itself.",
            "Matthews DR et al. HOMA model.",
        ),
        metadata(
            "estimated-average-glucose",
            "endocrinology",
            "Estimated average glucose",
            "HbA1c-derived average glucose estimate.",
            vec![number("hba1c_percent", "HbA1c (%)", 3.0, 20.0, None)],
            &["May be inaccurate when red-cell turnover or hemoglobin variants affect HbA1c."],
            "Use when HbA1c reliably reflects glycemia.",
            "Nathan DM et al. ADAG relationship.",
        ),
        metadata(
            "corrected-sodium",
            "endocrinology",
            "Corrected sodium in hyperglycemia",
            "Sodium corrected by 1.6 mmol/L per 100 mg/dL glucose above 100.",
            vec![
                number(
                    "sodium_mmol_l",
                    "Measured sodium (mmol/L)",
                    80.0,
                    200.0,
                    None,
                ),
                number("glucose_mg_dl", "Glucose (mg/dL)", 20.0, 2000.0, None),
            ],
            &["The correction factor is an approximation and differs in severe hyperglycemia."],
            "Hyperglycemia with glucose reported in mg/dL.",
            "Katz correction factor.",
        ),
        metadata(
            "anc",
            "hematology",
            "Absolute neutrophil count",
            "WBC multiplied by segmented neutrophil and band fraction.",
            vec![
                number("wbc", "WBC (×10⁹/L)", 0.01, 200.0, None),
                number("neutrophils_percent", "Neutrophils (%)", 0.0, 100.0, None),
                number("bands_percent", "Bands (%)", 0.0, 100.0, Some(0.0)),
            ],
            &[],
            "CBC differential; entered percentages must sum to no more than 100.",
            "ANC = WBC × (neutrophils + bands) / 100.",
        ),
        metadata(
            "mentzer-index",
            "hematology",
            "Mentzer index",
            "MCV divided by red blood cell count.",
            vec![
                number("mcv_fl", "MCV (fL)", 20.0, 150.0, None),
                number("rbc", "RBC (×10¹²/L)", 0.1, 10.0, None),
            ],
            &["Screening heuristic only; iron studies and hemoglobin analysis establish etiology."],
            "Microcytic anemia.",
            "Mentzer WC Jr. index.",
        ),
        metadata(
            "wells-pe",
            "hematology",
            "Wells score for pulmonary embolism",
            "Clinical pretest-probability score for pulmonary embolism.",
            vec![
                checkbox("clinical_dvt", "Clinical signs of DVT"),
                checkbox(
                    "pe_more_likely",
                    "PE more likely than alternative diagnosis",
                ),
                number("heart_rate", "Heart rate (/min)", 10.0, 300.0, None),
                checkbox(
                    "immobilization_or_surgery",
                    "Immobilization ≥3 days or surgery in prior 4 weeks",
                ),
                checkbox("prior_dvt_pe", "Prior DVT/PE"),
                checkbox("hemoptysis", "Hemoptysis"),
                checkbox("malignancy", "Active/recently treated malignancy"),
            ],
            &["Use the interpretation pathway validated by the local diagnostic protocol."],
            "Hemodynamically stable adults with suspected pulmonary embolism.",
            "Wells PS et al. PE clinical model.",
        ),
        metadata(
            "estimated-due-date",
            "obstetrics",
            "Estimated due date (Naegele)",
            "Adds 280 days to the first day of the last menstrual period.",
            vec![
                number("lmp_year", "LMP year", 1900.0, 2200.0, None),
                number("lmp_month", "LMP month", 1.0, 12.0, None),
                number("lmp_day", "LMP day", 1.0, 31.0, None),
            ],
            &["Cycle length and ultrasound dating can change the best obstetric estimate."],
            "Regular 28-day cycle with known first day of last menstrual period.",
            "Naegele rule: LMP + 280 days.",
        ),
        metadata(
            "bishop-score",
            "obstetrics",
            "Bishop score",
            "Cervical examination score before induction.",
            vec![
                number("dilation_cm", "Dilation (cm)", 0.0, 10.0, None),
                number("effacement_percent", "Effacement (%)", 0.0, 100.0, None),
                select(
                    "station",
                    "Fetal station",
                    &[
                        ("minus-3", "−3"),
                        ("minus-2", "−2"),
                        ("minus-1-zero", "−1 or 0"),
                        ("plus-1-2", "+1 or +2"),
                    ],
                    "minus-3",
                ),
                select(
                    "consistency",
                    "Cervical consistency",
                    &[("firm", "Firm"), ("medium", "Medium"), ("soft", "Soft")],
                    "firm",
                ),
                select(
                    "position",
                    "Cervical position",
                    &[
                        ("posterior", "Posterior"),
                        ("mid", "Mid-position"),
                        ("anterior", "Anterior"),
                    ],
                    "posterior",
                ),
            ],
            &["Exam findings are subjective and management depends on the full obstetric context."],
            "Term obstetric patients being assessed for induction.",
            "Bishop EH (1964).",
        ),
        metadata(
            "bedside-schwartz",
            "pediatrics",
            "Bedside Schwartz eGFR",
            "Height and creatinine pediatric eGFR estimate.",
            vec![
                number("height_cm", "Height (cm)", 20.0, 220.0, None),
                number(
                    "creatinine_mg_dl",
                    "Serum creatinine (mg/dL)",
                    0.1,
                    15.0,
                    None,
                ),
            ],
            &[],
            "Children and adolescents with stable creatinine measured using an IDMS-traceable assay.",
            "Schwartz GJ et al. 2009 bedside equation.",
        ),
        metadata(
            "holliday-segar",
            "pediatrics",
            "Holliday–Segar maintenance fluids",
            "Weight-based daily maintenance water estimate.",
            vec![number("weight_kg", "Weight (kg)", 0.1, 200.0, None)],
            &[
                "Maintenance estimates do not include deficits, ongoing losses, resuscitation, or disease-specific restriction.",
            ],
            "Children without conditions requiring a modified fluid strategy.",
            "Holliday MA, Segar WE (1957).",
        ),
    ]
}

fn registry() -> CalculatorRegistry {
    let definitions = calculators();
    let groups = [
        ("general-practice", "General practice"),
        ("cardiology", "Cardiology"),
        ("nephrology", "Nephrology"),
        ("pulmonology", "Pulmonology"),
        ("emergency-icu", "Emergency & ICU"),
        ("hepatology", "Hepatology"),
        ("endocrinology", "Endocrinology"),
        ("hematology", "Hematology"),
        ("obstetrics", "Obstetrics"),
        ("pediatrics", "Pediatrics"),
    ]
    .into_iter()
    .map(|(id, title)| CalculatorGroup {
        id: id.into(),
        title: ru(title),
        calculators: definitions
            .iter()
            .filter(|calculator| calculator.group == id)
            .cloned()
            .collect(),
    })
    .collect();
    CalculatorRegistry {
        version: REGISTRY_VERSION.into(),
        count: definitions.len(),
        groups,
    }
}

pub async fn list_calculators(user: AuthUser) -> Result<Json<CalculatorRegistry>, AppError> {
    user.require_password_changed()?;
    Ok(Json(registry()))
}

pub async fn calculate(
    user: AuthUser,
    Path(id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<CalculatorResult>, AppError> {
    user.require_password_changed()?;
    Ok(Json(dispatch(&id, &body)?))
}

fn object(body: &Value) -> Result<&Map<String, Value>, AppError> {
    body.as_object()
        .ok_or_else(|| bad("Тело запроса должно быть объектом JSON"))
}

fn n(body: &Value, id: &str, min: f64, max: f64) -> Result<f64, AppError> {
    let value = object(body)?
        .get(id)
        .and_then(Value::as_f64)
        .ok_or_else(|| bad(&format!("Поле '{id}' должно быть числом")))?;
    if !value.is_finite() || value < min || value > max {
        return Err(bad(&format!(
            "Поле '{id}' должно быть в диапазоне от {min} до {max}"
        )));
    }
    Ok(value)
}

fn integer(body: &Value, id: &str, min: i32, max: i32) -> Result<i32, AppError> {
    let value = n(body, id, min as f64, max as f64)?;
    if value.fract() != 0.0 {
        return Err(bad(&format!("Поле '{id}' должно быть целым числом")));
    }
    Ok(value as i32)
}

fn b(body: &Value, id: &str) -> Result<bool, AppError> {
    match object(body)?.get(id) {
        None => Ok(false),
        Some(Value::Bool(value)) => Ok(*value),
        _ => Err(bad(&format!(
            "Поле '{id}' должно быть логическим значением"
        ))),
    }
}

fn s<'a>(body: &'a Value, id: &str, allowed: &[&str]) -> Result<&'a str, AppError> {
    let value = object(body)?
        .get(id)
        .and_then(Value::as_str)
        .ok_or_else(|| bad(&format!("Поле '{id}' должно быть строкой")))?;
    if !allowed.contains(&value) {
        return Err(bad(&format!(
            "Поле '{id}' должно иметь одно из значений: {}",
            allowed.join(", ")
        )));
    }
    Ok(value)
}

fn bad(message: &str) -> AppError {
    AppError::BadRequest(message.into())
}

fn rounded(value: f64, digits: i32) -> Value {
    let scale = 10_f64.powi(digits);
    json!((value * scale).round() / scale)
}

fn result(
    value: Value,
    unit: &str,
    interpretation: impl Into<String>,
    details: Vec<String>,
    warnings: Vec<String>,
    reference: &str,
) -> CalculatorResult {
    CalculatorResult {
        value,
        unit: unit.into(),
        interpretation: interpretation.into(),
        details,
        warnings,
        reference: ru(reference),
    }
}

fn score_bool(body: &Value, id: &str, points: f64) -> Result<f64, AppError> {
    Ok(if b(body, id)? { points } else { 0.0 })
}

fn dispatch(id: &str, body: &Value) -> Result<CalculatorResult, AppError> {
    let r = match id {
        "bmi" => {
            let height_m = n(body, "height_cm", 50.0, 300.0)? / 100.0;
            let value = n(body, "weight_kg", 1.0, 500.0)? / height_m.powi(2);
            let interpretation = if value < 18.5 {
                "Ниже общепринятого диапазона для взрослых"
            } else if value < 25.0 {
                "В пределах общепринятого диапазона для взрослых"
            } else if value < 30.0 {
                "Диапазон избыточной массы тела"
            } else {
                "Диапазон ожирения"
            };
            result(
                rounded(value, 1),
                "кг/м²",
                interpretation,
                vec![],
                vec![],
                "Quetelet body mass index formula.",
            )
        }
        "bsa-du-bois" => {
            let h = n(body, "height_cm", 50.0, 300.0)?;
            let w = n(body, "weight_kg", 1.0, 500.0)?;
            result(
                rounded(0.007184 * h.powf(0.725) * w.powf(0.425), 2),
                "м²",
                "Расчётная площадь поверхности тела",
                vec![],
                vec![],
                "Du Bois D, Du Bois EF (1916).",
            )
        }
        "ideal-body-weight-devine" => {
            let sex = s(body, "sex", &["male", "female"])?;
            let inches_over_five_feet =
                (n(body, "height_cm", 120.0, 250.0)? / 2.54 - 60.0).max(0.0);
            let value = if sex == "male" { 50.0 } else { 45.5 } + 2.3 * inches_over_five_feet;
            result(
                rounded(value, 1),
                "кг",
                "Расчёт идеальной массы тела по росту",
                vec![],
                vec![],
                "Devine BJ (1974).",
            )
        }
        "mg-kg-dose" => {
            let value = n(body, "weight_kg", 0.1, 500.0)? * n(body, "dose_mg_kg", 0.0, 1000.0)?;
            result(rounded(value, 2), "мг", "Количество, арифметически рассчитанное по введённому назначению", vec![], vec!["Расчёт не рекомендует препарат, дозу, путь или интервал введения либо максимальную дозу.".into()], "Размерностный расчёт: кг × мг/кг.")
        }
        "mean-arterial-pressure" => {
            let sys = n(body, "systolic_bp", 30.0, 300.0)?;
            let dia = n(body, "diastolic_bp", 10.0, 200.0)?;
            if dia > sys {
                return Err(bad("Значение diastolic_bp не может превышать systolic_bp"));
            }
            result(
                rounded((sys + 2.0 * dia) / 3.0, 1),
                "мм рт. ст.",
                "Расчётное среднее артериальное давление",
                vec![],
                vec![],
                "MAP approximation.",
            )
        }
        "cha2ds2-vasc" => {
            let age = n(body, "age", 18.0, 120.0)?;
            let sex = s(body, "sex", &["male", "female"])?;
            let value = score_bool(body, "heart_failure", 1.0)?
                + score_bool(body, "hypertension", 1.0)?
                + score_bool(body, "diabetes", 1.0)?
                + score_bool(body, "stroke_tia", 2.0)?
                + score_bool(body, "vascular_disease", 1.0)?
                + if age >= 75.0 {
                    2.0
                } else if age >= 65.0 {
                    1.0
                } else {
                    0.0
                }
                + if sex == "female" { 1.0 } else { 0.0 };
            result(
                json!(value as i32),
                "баллы",
                if value >= 2.0 {
                    "Повышенная выраженность факторов риска инсульта"
                } else {
                    "Низкая выраженность факторов риска инсульта"
                },
                vec![],
                vec![],
                "CHA₂DS₂-VASc risk-factor scheme.",
            )
        }
        "has-bled" => {
            let ids = [
                "hypertension",
                "abnormal_renal",
                "abnormal_liver",
                "stroke",
                "bleeding",
                "labile_inr",
                "age_over_65",
                "drugs",
                "alcohol",
            ];
            let value = ids.iter().try_fold(0, |sum, id| {
                Ok::<_, AppError>(sum + i32::from(b(body, id)?))
            })?;
            result(
                json!(value),
                "баллы",
                if value >= 3 {
                    "Высокий балл риска кровотечения; оцените модифицируемые факторы"
                } else {
                    "Низкий балл риска кровотечения"
                },
                vec![],
                vec!["Не используйте только этот балл для отказа от антикоагуляции.".into()],
                "HAS-BLED score.",
            )
        }
        "qtc" => {
            let qt = n(body, "qt_ms", 100.0, 1000.0)?;
            let rr = 60.0 / n(body, "heart_rate", 20.0, 250.0)?;
            let formula = s(body, "formula", &["bazett", "fridericia"])?;
            let value = if formula == "bazett" {
                qt / rr.sqrt()
            } else {
                qt / rr.cbrt()
            };
            let formula_label = if formula == "bazett" {
                "Базетт"
            } else {
                "Фридеричиа"
            };
            result(
                rounded(value, 0),
                "мс",
                "Интервал QT, скорректированный по ЧСС",
                vec![format!("Формула: {formula_label}")],
                vec![],
                "Bazett or Fridericia correction.",
            )
        }
        "cockcroft-gault" => {
            let age = n(body, "age", 18.0, 120.0)?;
            let sex = s(body, "sex", &["male", "female"])?;
            let value = (140.0 - age) * n(body, "weight_kg", 20.0, 500.0)?
                / (72.0 * n(body, "creatinine_mg_dl", 0.1, 20.0)?)
                * if sex == "female" { 0.85 } else { 1.0 };
            result(
                rounded(value, 1),
                "мл/мин",
                "Расчётный клиренс креатинина",
                vec![],
                vec![],
                "Cockcroft–Gault equation.",
            )
        }
        "ckd-epi-2021" => {
            let age = n(body, "age", 18.0, 120.0)?;
            let scr = n(body, "creatinine_mg_dl", 0.1, 20.0)?;
            let sex = s(body, "sex", &["male", "female"])?;
            let (k, alpha, factor) = if sex == "female" {
                (0.7, -0.241, 1.012)
            } else {
                (0.9, -0.302, 1.0)
            };
            let ratio = scr / k;
            let value = 142.0
                * ratio.min(1.0).powf(alpha)
                * ratio.max(1.0).powf(-1.2)
                * 0.9938_f64.powf(age)
                * factor;
            result(
                rounded(value, 0),
                "мл/мин/1,73 м²",
                "Расчётная скорость клубочковой фильтрации",
                vec![],
                vec![],
                "2021 CKD-EPI creatinine equation.",
            )
        }
        "fena" => {
            let value = 100.0
                * n(body, "urine_sodium", 0.1, 500.0)?
                * n(body, "plasma_creatinine", 0.1, 1000.0)?
                / (n(body, "plasma_sodium", 80.0, 200.0)?
                    * n(body, "urine_creatinine", 0.1, 1000.0)?);
            result(
                rounded(value, 2),
                "%",
                if value < 1.0 {
                    "Низкая фракционная экскреция натрия"
                } else {
                    "Фракционная экскреция натрия не менее 1%"
                },
                vec![],
                vec![],
                "FENa formula.",
            )
        }
        "feurea" => {
            let value = 100.0
                * n(body, "urine_urea", 0.1, 5000.0)?
                * n(body, "plasma_creatinine", 0.1, 1000.0)?
                / (n(body, "plasma_urea", 0.1, 500.0)? * n(body, "urine_creatinine", 0.1, 1000.0)?);
            result(
                rounded(value, 2),
                "%",
                if value < 35.0 {
                    "Низкая фракционная экскреция мочевины"
                } else {
                    "Фракционная экскреция мочевины не менее 35%"
                },
                vec![],
                vec![],
                "FEUrea formula.",
            )
        }
        "curb-65" => {
            let sys = n(body, "systolic_bp", 30.0, 300.0)?;
            let dia = n(body, "diastolic_bp", 10.0, 200.0)?;
            if dia > sys {
                return Err(bad("Значение diastolic_bp не может превышать systolic_bp"));
            }
            let value = i32::from(b(body, "confusion")?)
                + i32::from(n(body, "urea_mmol_l", 0.1, 100.0)? > 7.0)
                + i32::from(n(body, "respiratory_rate", 1.0, 80.0)? >= 30.0)
                + i32::from(sys < 90.0 || dia <= 60.0)
                + i32::from(n(body, "age", 18.0, 120.0)? >= 65.0);
            result(
                json!(value),
                "баллы",
                match value {
                    0..=1 => "Низкий балл",
                    2 => "Промежуточный балл",
                    _ => "Высокий балл",
                },
                vec![],
                vec![],
                "CURB-65.",
            )
        }
        "pf-ratio" => {
            let value = n(body, "pao2_mm_hg", 20.0, 700.0)?
                / (n(body, "fio2_percent", 21.0, 100.0)? / 100.0);
            result(
                rounded(value, 0),
                "мм рт. ст.",
                if value <= 300.0 {
                    "Диапазон нарушения оксигенации"
                } else {
                    "Отношение P/F выше 300"
                },
                vec![],
                vec![],
                "PaO₂/FiO₂.",
            )
        }
        "aa-gradient" => {
            let age = n(body, "age", 0.0, 120.0)?;
            let fio2 = n(body, "fio2_percent", 21.0, 100.0)? / 100.0;
            let value = fio2 * (n(body, "atmospheric_pressure", 300.0, 800.0)? - 47.0)
                - n(body, "paco2_mm_hg", 5.0, 150.0)? / 0.8
                - n(body, "pao2_mm_hg", 10.0, 700.0)?;
            let expected = age / 4.0 + 4.0;
            result(
                rounded(value, 1),
                "мм рт. ст.",
                if value <= expected {
                    "В пределах ожидаемого возрастного диапазона"
                } else {
                    "Выше ожидаемого возрастного диапазона"
                },
                vec![format!(
                    "Ожидаемая верхняя граница с поправкой на возраст: {:.1} мм рт. ст.",
                    expected
                )],
                vec![],
                "Alveolar gas equation.",
            )
        }
        "smoking-pack-years" => {
            let value = n(body, "packs_per_day", 0.0, 10.0)? * n(body, "years_smoked", 0.0, 100.0)?;
            result(
                rounded(value, 1),
                "пачка-лет",
                "Расчётная суммарная табачная нагрузка",
                vec![],
                vec![],
                "Packs/day × years.",
            )
        }
        "glasgow-coma-scale" => {
            let parse_component = |field, allowed: &[&str]| -> Result<i32, AppError> {
                s(body, field, allowed)?
                    .parse()
                    .map_err(|_| bad("Недопустимый компонент GCS"))
            };
            let eye = parse_component("eye", &["1", "2", "3", "4"])?;
            let verbal = parse_component("verbal", &["1", "2", "3", "4", "5"])?;
            let motor = parse_component("motor", &["1", "2", "3", "4", "5", "6"])?;
            let value = eye + verbal + motor;
            result(
                json!(value),
                "баллы",
                if value <= 8 {
                    "Диапазон тяжёлого нарушения сознания"
                } else if value <= 12 {
                    "Диапазон умеренного нарушения сознания"
                } else {
                    "Лёгкое нарушение сознания или его отсутствие"
                },
                vec![format!("E{eye} V{verbal} M{motor}")],
                vec![],
                "Glasgow Coma Scale.",
            )
        }
        "qsofa" => {
            let value = i32::from(n(body, "respiratory_rate", 1.0, 80.0)? >= 22.0)
                + i32::from(n(body, "systolic_bp", 30.0, 300.0)? <= 100.0)
                + i32::from(b(body, "altered_mentation")?);
            result(
                json!(value),
                "баллы",
                if value >= 2 {
                    "Не менее двух критериев qSOFA"
                } else {
                    "Менее двух критериев qSOFA"
                },
                vec![],
                vec!["Не является самостоятельным скрининговым или диагностическим тестом на сепсис.".into()],
                "Sepsis-3 qSOFA.",
            )
        }
        "sirs" => {
            let temp = n(body, "temperature_c", 25.0, 45.0)?;
            let wbc = n(body, "wbc", 0.01, 200.0)?;
            let value = i32::from(temp > 38.0 || temp < 36.0)
                + i32::from(n(body, "heart_rate", 10.0, 300.0)? > 90.0)
                + i32::from(
                    n(body, "respiratory_rate", 1.0, 80.0)? > 20.0
                        || n(body, "paco2_mm_hg", 5.0, 150.0)? < 32.0,
                )
                + i32::from(
                    wbc > 12.0 || wbc < 4.0 || n(body, "bands_percent", 0.0, 100.0)? > 10.0,
                );
            result(
                json!(value),
                "критерии",
                if value >= 2 {
                    "Не менее двух критериев SIRS"
                } else {
                    "Менее двух критериев SIRS"
                },
                vec![],
                vec!["SIRS неспецифичен и не является современным определением сепсиса.".into()],
                "ACCP/SCCM SIRS criteria.",
            )
        }
        "shock-index" => {
            let value = n(body, "heart_rate", 10.0, 300.0)? / n(body, "systolic_bp", 30.0, 300.0)?;
            result(
                rounded(value, 2),
                "отношение",
                if value >= 0.9 {
                    "Повышенный шоковый индекс"
                } else {
                    "Шоковый индекс ниже 0,9"
                },
                vec![],
                vec![],
                "Allgöwer shock index.",
            )
        }
        "child-pugh" => {
            let bilirubin = n(body, "bilirubin_mg_dl", 0.1, 50.0)?;
            let albumin = n(body, "albumin_g_dl", 0.5, 6.0)?;
            let inr = n(body, "inr", 0.5, 10.0)?;
            let ascites = s(body, "ascites", &["none", "mild", "moderate-severe"])?;
            let enceph = s(body, "encephalopathy", &["none", "grade-1-2", "grade-3-4"])?;
            let value = if bilirubin < 2.0 {
                1
            } else if bilirubin <= 3.0 {
                2
            } else {
                3
            } + if albumin > 3.5 {
                1
            } else if albumin >= 2.8 {
                2
            } else {
                3
            } + if inr < 1.7 {
                1
            } else if inr <= 2.3 {
                2
            } else {
                3
            } + match ascites {
                "none" => 1,
                "mild" => 2,
                _ => 3,
            } + match enceph {
                "none" => 1,
                "grade-1-2" => 2,
                _ => 3,
            };
            let class = if value <= 6 {
                "A"
            } else if value <= 9 {
                "B"
            } else {
                "C"
            };
            result(
                json!(value),
                "баллы",
                format!("Класс Чайлд—Пью {class}"),
                vec![],
                vec![],
                "Child–Pugh score.",
            )
        }
        "meld-3" => {
            let bilirubin = n(body, "bilirubin_mg_dl", 0.1, 80.0)?.max(1.0);
            let inr = n(body, "inr", 0.5, 20.0)?.max(1.0);
            let mut creatinine = n(body, "creatinine_mg_dl", 0.1, 20.0)?.clamp(1.0, 3.0);
            if b(body, "dialysis_twice_weekly")? {
                creatinine = 3.0;
            }
            let sodium = n(body, "sodium_mmol_l", 100.0, 180.0)?.clamp(125.0, 137.0);
            let albumin = n(body, "albumin_g_dl", 0.5, 6.0)?.clamp(1.5, 3.5);
            let female = s(body, "sex", &["male", "female"])? == "female";
            let value = 1.33 * f64::from(female) + 4.56 * bilirubin.ln() + 0.82 * (137.0 - sodium)
                - 0.24 * (137.0 - sodium) * bilirubin.ln()
                + 9.09 * inr.ln()
                + 11.14 * creatinine.ln()
                + 1.85 * (3.5 - albumin)
                - 1.83 * (3.5 - albumin) * creatinine.ln()
                + 6.0;
            let score = value.round().clamp(6.0, 40.0) as i32;
            result(
                json!(score),
                "баллы",
                "Балл MELD 3.0 с официальными ограничениями лабораторных значений",
                vec![format!(
                    "Использованы ограниченные значения: билирубин {bilirubin:.2}, МНО {inr:.2}, креатинин {creatinine:.2}, натрий {sodium:.1}, альбумин {albumin:.1}"
                )],
                vec![],
                "Kim WR et al. MELD 3.0.",
            )
        }
        "fib-4" => {
            let value = n(body, "age", 18.0, 120.0)? * n(body, "ast_u_l", 1.0, 10000.0)?
                / (n(body, "platelets", 1.0, 2000.0)? * n(body, "alt_u_l", 1.0, 10000.0)?.sqrt());
            result(
                rounded(value, 2),
                "индекс",
                "Расчёт FIB-4; применяйте пороги соответствующего клинического маршрута",
                vec![],
                vec![],
                "FIB-4 formula.",
            )
        }
        "homa-ir" => {
            let value = n(body, "glucose_mmol_l", 1.0, 50.0)?
                * n(body, "insulin_miu_l", 0.1, 1000.0)?
                / 22.5;
            result(
                rounded(value, 2),
                "индекс",
                "Расчёт HOMA-IR",
                vec![],
                vec![],
                "HOMA-IR formula.",
            )
        }
        "estimated-average-glucose" => {
            let value = 28.7 * n(body, "hba1c_percent", 3.0, 20.0)? - 46.7;
            result(
                rounded(value, 0),
                "мг/дл",
                "Расчётная средняя глюкоза по HbA1c",
                vec![format!("Эквивалент: {:.1} ммоль/л", value / 18.0)],
                vec![],
                "ADAG relationship.",
            )
        }
        "corrected-sodium" => {
            let sodium = n(body, "sodium_mmol_l", 80.0, 200.0)?;
            let glucose = n(body, "glucose_mg_dl", 20.0, 2000.0)?;
            let value = sodium + 1.6 * ((glucose - 100.0).max(0.0) / 100.0);
            result(
                rounded(value, 1),
                "ммоль/л",
                "Расчёт натрия с поправкой на глюкозу",
                vec![],
                vec![],
                "Katz correction factor.",
            )
        }
        "anc" => {
            let neutrophils = n(body, "neutrophils_percent", 0.0, 100.0)?;
            let bands = n(body, "bands_percent", 0.0, 100.0)?;
            if neutrophils + bands > 100.0 {
                return Err(bad(
                    "Сумма neutrophils_percent и bands_percent не может превышать 100",
                ));
            }
            let value = n(body, "wbc", 0.01, 200.0)? * (neutrophils + bands) / 100.0;
            result(
                rounded(value, 2),
                "×10⁹/л",
                if value < 0.5 {
                    "Диапазон тяжёлой нейтропении"
                } else if value < 1.0 {
                    "Диапазон умеренной нейтропении"
                } else if value < 1.5 {
                    "Диапазон лёгкой нейтропении"
                } else {
                    "Вне общепринятого диапазона нейтропении"
                },
                vec![],
                vec![],
                "ANC formula.",
            )
        }
        "mentzer-index" => {
            let value = n(body, "mcv_fl", 20.0, 150.0)? / n(body, "rbc", 0.1, 10.0)?;
            result(
                rounded(value, 1),
                "индекс",
                if value < 13.0 {
                    "По этому ориентиру картина больше соответствует носительству талассемии"
                } else {
                    "По этому ориентиру картина больше соответствует дефициту железа"
                },
                vec![],
                vec![],
                "Mentzer index.",
            )
        }
        "wells-pe" => {
            let heart_rate = n(body, "heart_rate", 10.0, 300.0)?;
            let value = score_bool(body, "clinical_dvt", 3.0)?
                + score_bool(body, "pe_more_likely", 3.0)?
                + if heart_rate > 100.0 { 1.5 } else { 0.0 }
                + score_bool(body, "immobilization_or_surgery", 1.5)?
                + score_bool(body, "prior_dvt_pe", 1.5)?
                + score_bool(body, "hemoptysis", 1.0)?
                + score_bool(body, "malignancy", 1.0)?;
            result(
                rounded(value, 1),
                "баллы",
                if value > 4.0 {
                    "ТЭЛА вероятна по двухуровневой модели"
                } else {
                    "ТЭЛА маловероятна по двухуровневой модели"
                },
                vec![],
                vec![],
                "Wells PE score.",
            )
        }
        "estimated-due-date" => {
            let year = integer(body, "lmp_year", 1900, 2200)?;
            let month = integer(body, "lmp_month", 1, 12)? as u32;
            let day = integer(body, "lmp_day", 1, 31)? as u32;
            let lmp = NaiveDate::from_ymd_opt(year, month, day).ok_or_else(|| {
                bad("Компоненты даты последней менструации не образуют допустимую дату")
            })?;
            let due = lmp
                .checked_add_signed(Duration::days(280))
                .ok_or_else(|| bad("Расчётная дата выходит за допустимый диапазон"))?;
            result(
                json!(due.format("%Y-%m-%d").to_string()),
                "дата",
                "Предполагаемая дата родов",
                vec![format!("Последняя менструация: {}", lmp.format("%Y-%m-%d"))],
                vec![],
                "Naegele rule.",
            )
        }
        "bishop-score" => {
            let dilation = n(body, "dilation_cm", 0.0, 10.0)?;
            let effacement = n(body, "effacement_percent", 0.0, 100.0)?;
            let station = s(
                body,
                "station",
                &["minus-3", "minus-2", "minus-1-zero", "plus-1-2"],
            )?;
            let consistency = s(body, "consistency", &["firm", "medium", "soft"])?;
            let position = s(body, "position", &["posterior", "mid", "anterior"])?;
            let value = if dilation == 0.0 {
                0
            } else if dilation <= 2.0 {
                1
            } else if dilation <= 4.0 {
                2
            } else {
                3
            } + if effacement <= 30.0 {
                0
            } else if effacement <= 50.0 {
                1
            } else if effacement <= 70.0 {
                2
            } else {
                3
            } + match station {
                "minus-3" => 0,
                "minus-2" => 1,
                "minus-1-zero" => 2,
                _ => 3,
            } + match consistency {
                "firm" => 0,
                "medium" => 1,
                _ => 2,
            } + match position {
                "posterior" => 0,
                "mid" => 1,
                _ => 2,
            };
            result(
                json!(value),
                "баллы",
                if value >= 8 {
                    "В целом благоприятная оценка шейки матки"
                } else {
                    "Менее благоприятная оценка шейки матки"
                },
                vec![],
                vec![],
                "Bishop score.",
            )
        }
        "bedside-schwartz" => {
            let value = 0.413 * n(body, "height_cm", 20.0, 220.0)?
                / n(body, "creatinine_mg_dl", 0.1, 15.0)?;
            result(
                rounded(value, 0),
                "мл/мин/1,73 м²",
                "Расчётная СКФ у ребёнка",
                vec![],
                vec![],
                "2009 bedside Schwartz equation.",
            )
        }
        "holliday-segar" => {
            let weight = n(body, "weight_kg", 0.1, 200.0)?;
            let value = if weight <= 10.0 {
                weight * 100.0
            } else if weight <= 20.0 {
                1000.0 + (weight - 10.0) * 50.0
            } else {
                1500.0 + (weight - 20.0) * 20.0
            };
            result(
                rounded(value, 0),
                "мл/сут",
                "Расчётный суточный объём поддерживающей жидкости",
                vec![format!("Почасовой эквивалент: {:.1} мл/ч", value / 24.0)],
                vec![],
                "Holliday–Segar method.",
            )
        }
        _ => {
            return Err(AppError::NotFound(format!(
                "Неизвестный калькулятор '{id}'"
            )));
        }
    };
    Ok(r)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(id: &str, body: Value) -> Value {
        dispatch(id, &body).unwrap().value
    }

    #[test]
    fn registry_has_all_groups_and_33_calculators() {
        let registry = registry();
        assert_eq!(registry.groups.len(), 10);
        assert_eq!(registry.count, 33);
        assert!(
            registry
                .groups
                .iter()
                .all(|group| !group.calculators.is_empty())
        );
        assert!(
            !calculators()
                .iter()
                .any(|calculator| calculator.id == "score")
        );
    }

    #[test]
    fn registry_and_api_messages_are_localized() {
        let contains_cyrillic = |text: &str| {
            text.chars()
                .any(|character| matches!(character, '\u{0400}'..='\u{04ff}'))
        };
        let registry = registry();
        for group in registry.groups {
            assert!(contains_cyrillic(&group.title), "group {}", group.id);
            for calculator in group.calculators {
                assert!(
                    contains_cyrillic(&calculator.description),
                    "description {}",
                    calculator.id
                );
                assert!(
                    contains_cyrillic(&calculator.applicability),
                    "applicability {}",
                    calculator.id
                );
                assert!(
                    calculator
                        .warnings
                        .iter()
                        .all(|warning| contains_cyrillic(warning)),
                    "warnings {}",
                    calculator.id
                );
            }
        }

        let bmi = dispatch("bmi", &json!({"height_cm": 180, "weight_kg": 81})).unwrap();
        assert!(contains_cyrillic(&bmi.interpretation));
        let error = dispatch("bmi", &json!({"height_cm": "180", "weight_kg": 81})).unwrap_err();
        assert!(matches!(error, AppError::BadRequest(message) if contains_cyrillic(&message)));
        assert!(
            matches!(dispatch("score", &json!({})), Err(AppError::NotFound(message)) if contains_cyrillic(&message))
        );
    }

    #[test]
    fn golden_calculator_results() {
        let cases = [
            (
                "bmi",
                json!({"height_cm": 180, "weight_kg": 81}),
                json!(25.0),
            ),
            (
                "bsa-du-bois",
                json!({"height_cm": 180, "weight_kg": 80}),
                json!(2.0),
            ),
            (
                "ideal-body-weight-devine",
                json!({"sex":"male","height_cm":177.8}),
                json!(73.0),
            ),
            (
                "mg-kg-dose",
                json!({"weight_kg":20,"dose_mg_kg":5}),
                json!(100.0),
            ),
            (
                "mean-arterial-pressure",
                json!({"systolic_bp":120,"diastolic_bp":60}),
                json!(80.0),
            ),
            (
                "cha2ds2-vasc",
                json!({"age":76,"sex":"female","heart_failure":true}),
                json!(4),
            ),
            (
                "has-bled",
                json!({"hypertension":true,"abnormal_renal":true,"alcohol":true}),
                json!(3),
            ),
            (
                "qtc",
                json!({"qt_ms":400,"heart_rate":60,"formula":"bazett"}),
                json!(400.0),
            ),
            (
                "cockcroft-gault",
                json!({"age":40,"weight_kg":72,"creatinine_mg_dl":1,"sex":"male"}),
                json!(100.0),
            ),
            (
                "ckd-epi-2021",
                json!({"age":40,"creatinine_mg_dl":1,"sex":"male"}),
                json!(98.0),
            ),
            (
                "fena",
                json!({"urine_sodium":40,"plasma_sodium":140,"urine_creatinine":100,"plasma_creatinine":2}),
                json!(0.57),
            ),
            (
                "feurea",
                json!({"urine_urea":200,"plasma_urea":20,"urine_creatinine":100,"plasma_creatinine":2}),
                json!(20.0),
            ),
            (
                "curb-65",
                json!({"confusion":true,"urea_mmol_l":8,"respiratory_rate":30,"systolic_bp":100,"diastolic_bp":60,"age":65}),
                json!(5),
            ),
            (
                "pf-ratio",
                json!({"pao2_mm_hg":80,"fio2_percent":40}),
                json!(200.0),
            ),
            (
                "aa-gradient",
                json!({"age":40,"fio2_percent":21,"paco2_mm_hg":40,"pao2_mm_hg":100,"atmospheric_pressure":760}),
                json!(-0.3),
            ),
            (
                "smoking-pack-years",
                json!({"packs_per_day":1.5,"years_smoked":20}),
                json!(30.0),
            ),
            (
                "glasgow-coma-scale",
                json!({"eye":"4","verbal":"5","motor":"6"}),
                json!(15),
            ),
            (
                "qsofa",
                json!({"respiratory_rate":22,"systolic_bp":100,"altered_mentation":true}),
                json!(3),
            ),
            (
                "sirs",
                json!({"temperature_c":39,"heart_rate":100,"respiratory_rate":24,"paco2_mm_hg":40,"wbc":14,"bands_percent":0}),
                json!(4),
            ),
            (
                "shock-index",
                json!({"heart_rate":120,"systolic_bp":100}),
                json!(1.2),
            ),
            (
                "child-pugh",
                json!({"bilirubin_mg_dl":1,"albumin_g_dl":4,"inr":1,"ascites":"none","encephalopathy":"none"}),
                json!(5),
            ),
            (
                "meld-3",
                json!({"bilirubin_mg_dl":1,"inr":1,"creatinine_mg_dl":1,"sodium_mmol_l":137,"albumin_g_dl":3.5,"sex":"male"}),
                json!(6),
            ),
            (
                "fib-4",
                json!({"age":50,"ast_u_l":40,"alt_u_l":25,"platelets":200}),
                json!(2.0),
            ),
            (
                "homa-ir",
                json!({"glucose_mmol_l":5,"insulin_miu_l":9}),
                json!(2.0),
            ),
            (
                "estimated-average-glucose",
                json!({"hba1c_percent":7}),
                json!(154.0),
            ),
            (
                "corrected-sodium",
                json!({"sodium_mmol_l":130,"glucose_mg_dl":600}),
                json!(138.0),
            ),
            (
                "anc",
                json!({"wbc":2,"neutrophils_percent":40,"bands_percent":10}),
                json!(1.0),
            ),
            ("mentzer-index", json!({"mcv_fl":72,"rbc":6}), json!(12.0)),
            (
                "wells-pe",
                json!({"clinical_dvt":true,"pe_more_likely":true,"heart_rate":110,"prior_dvt_pe":true}),
                json!(9.0),
            ),
            (
                "estimated-due-date",
                json!({"lmp_year":2026,"lmp_month":1,"lmp_day":1}),
                json!("2026-10-08"),
            ),
            (
                "bishop-score",
                json!({"dilation_cm":5,"effacement_percent":80,"station":"plus-1-2","consistency":"soft","position":"anterior"}),
                json!(13),
            ),
            (
                "bedside-schwartz",
                json!({"height_cm":100,"creatinine_mg_dl":0.5}),
                json!(83.0),
            ),
            ("holliday-segar", json!({"weight_kg":25}), json!(1600.0)),
        ];
        for (id, body, expected) in cases {
            assert_eq!(value(id, body), expected, "golden mismatch for {id}");
        }
    }

    #[test]
    fn rejects_invalid_types_ranges_and_cross_field_values() {
        assert!(dispatch("bmi", &json!({"height_cm":"180","weight_kg":80})).is_err());
        assert!(dispatch("bmi", &json!({"height_cm":0,"weight_kg":80})).is_err());
        assert!(
            dispatch(
                "cockcroft-gault",
                &json!({"age":40,"weight_kg":70,"creatinine_mg_dl":1,"sex":"other"})
            )
            .is_err()
        );
        assert!(
            dispatch(
                "anc",
                &json!({"wbc":2,"neutrophils_percent":90,"bands_percent":20})
            )
            .is_err()
        );
        assert!(
            dispatch(
                "estimated-due-date",
                &json!({"lmp_year":2026,"lmp_month":2,"lmp_day":30})
            )
            .is_err()
        );
        assert!(matches!(
            dispatch("score", &json!({})),
            Err(AppError::NotFound(_))
        ));
    }
}
