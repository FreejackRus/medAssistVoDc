use axum::Json;
use serde::{Deserialize, Serialize};

use crate::{auth::AuthUser, error::AppError};

#[derive(Serialize)]
pub struct CalculatorResult {
    result: f64,
    unit: String,
    interpretation: String,
}

// --- BMI ---

#[derive(Deserialize)]
pub struct BMIRequest {
    height_cm: f64,
    weight_kg: f64,
}

pub async fn calc_bmi(
    user: AuthUser,
    Json(body): Json<BMIRequest>,
) -> Result<Json<CalculatorResult>, AppError> {
    user.require_password_changed()?;
    if body.height_cm <= 0.0 || body.weight_kg <= 0.0 {
        return Err(AppError::BadRequest("Values must be positive".into()));
    }
    let h_m = body.height_cm / 100.0;
    let bmi = body.weight_kg / (h_m * h_m);

    let interp = match bmi {
        b if b < 18.5 => "Недостаточная масса тела",
        b if b < 25.0 => "Нормальная масса тела",
        b if b < 30.0 => "Избыточная масса тела (предожирение)",
        b if b < 35.0 => "Ожирение I степени",
        b if b < 40.0 => "Ожирение II степени",
        _ => "Ожирение III степени (морбидное)",
    };

    Ok(Json(CalculatorResult {
        result: (bmi * 10.0).round() / 10.0,
        unit: "кг/м²".into(),
        interpretation: interp.into(),
    }))
}

// --- Creatinine Clearance (Cockcroft-Gault) ---

#[derive(Deserialize)]
pub struct CreatinineRequest {
    age: u32,
    weight_kg: f64,
    creatinine: f64,
    gender: String,
}

pub async fn calc_creatinine(
    user: AuthUser,
    Json(body): Json<CreatinineRequest>,
) -> Result<Json<CalculatorResult>, AppError> {
    user.require_password_changed()?;
    if body.age == 0 || body.age > 150 {
        return Err(AppError::BadRequest("Age must be 1-150".into()));
    }
    if body.weight_kg <= 0.0 || body.weight_kg > 500.0 {
        return Err(AppError::BadRequest("Weight must be 0-500 kg".into()));
    }
    if body.creatinine <= 0.0 || body.creatinine > 2000.0 {
        return Err(AppError::BadRequest(
            "Creatinine must be 0-2000 µmol/L".into(),
        ));
    }
    let factor = if body.gender == "female" { 1.04 } else { 1.23 };
    let ccr = ((140.0 - body.age as f64) * body.weight_kg * factor) / body.creatinine;

    let interp = match ccr {
        c if c > 90.0 => "Нормальная функция почек",
        c if c > 60.0 => "Легкое снижение функции почек",
        c if c > 30.0 => "Умеренное снижение функции почек",
        c if c > 15.0 => "Тяжелое снижение функции почек",
        _ => "Терминальная почечная недостаточность",
    };

    Ok(Json(CalculatorResult {
        result: (ccr * 10.0).round() / 10.0,
        unit: "мл/мин".into(),
        interpretation: interp.into(),
    }))
}

// --- BSA (Du Bois) ---

#[derive(Deserialize)]
pub struct BSARequest {
    height_cm: f64,
    weight_kg: f64,
}

pub async fn calc_bsa(
    user: AuthUser,
    Json(body): Json<BSARequest>,
) -> Result<Json<CalculatorResult>, AppError> {
    user.require_password_changed()?;
    if body.height_cm <= 0.0
        || body.height_cm > 300.0
        || body.weight_kg <= 0.0
        || body.weight_kg > 500.0
    {
        return Err(AppError::BadRequest(
            "Height must be 0-300 cm, weight 0-500 kg".into(),
        ));
    }
    let bsa = 0.007184 * body.height_cm.powf(0.725) * body.weight_kg.powf(0.425);
    Ok(Json(CalculatorResult {
        result: (bsa * 100.0).round() / 100.0,
        unit: "м²".into(),
        interpretation: format!("Площадь поверхности тела: {:.2} м²", bsa),
    }))
}

// --- Dosage ---

#[derive(Deserialize)]
pub struct DosageRequest {
    weight_kg: f64,
    dose_per_kg: f64,
    #[serde(default = "default_frequency")]
    frequency: u32,
}

fn default_frequency() -> u32 {
    1
}

pub async fn calc_dosage(
    user: AuthUser,
    Json(body): Json<DosageRequest>,
) -> Result<Json<CalculatorResult>, AppError> {
    user.require_password_changed()?;
    if body.weight_kg <= 0.0 || body.weight_kg > 500.0 {
        return Err(AppError::BadRequest("Weight must be 0-500 kg".into()));
    }
    if body.dose_per_kg <= 0.0 || body.dose_per_kg > 1000.0 {
        return Err(AppError::BadRequest("Dose must be 0-1000 mg/kg".into()));
    }
    let freq = if body.frequency > 0 && body.frequency <= 24 {
        body.frequency
    } else {
        1
    };
    let total = body.weight_kg * body.dose_per_kg;
    let per_dose = total / freq as f64;

    Ok(Json(CalculatorResult {
        result: (per_dose * 10.0).round() / 10.0,
        unit: "мг".into(),
        interpretation: format!(
            "Суточная доза: {:.1} мг. Разовая доза ({}x/сут): {:.1} мг",
            total, freq, per_dose
        ),
    }))
}

// --- SCORE (Educational) ---

#[derive(Deserialize)]
pub struct ScoreRequest {
    age: u32,
    gender: String,
    smoking: bool,
    cholesterol: f64,
    systolic_bp: u32,
}

pub async fn calc_score(
    user: AuthUser,
    Json(body): Json<ScoreRequest>,
) -> Result<Json<CalculatorResult>, AppError> {
    user.require_password_changed()?;
    // Educational simplified SCORE estimation — NOT for clinical use
    if body.age < 40 || body.age > 65 {
        return Err(AppError::BadRequest("SCORE age range is 40-65".into()));
    }
    if body.gender != "male" && body.gender != "female" {
        return Err(AppError::BadRequest(
            "Gender must be 'male' or 'female'".into(),
        ));
    }
    if body.cholesterol <= 0.0 || body.cholesterol > 20.0 {
        return Err(AppError::BadRequest(
            "Cholesterol must be 0-20 mmol/L".into(),
        ));
    }
    if body.systolic_bp < 80 || body.systolic_bp > 300 {
        return Err(AppError::BadRequest(
            "Systolic BP must be 80-300 mmHg".into(),
        ));
    }
    let base: f64 = if body.gender == "male" { 1.5 } else { 1.0 };
    let age_factor = (body.age as f64 - 40.0).max(0.0) * 0.15;
    let chol_factor = (body.cholesterol - 4.0).max(0.0) * 0.5;
    let bp_factor = (body.systolic_bp as f64 - 120.0).max(0.0) * 0.02;
    let smoke_factor = if body.smoking { 2.0 } else { 1.0 };
    let risk = (base * (1.0 + age_factor) * (1.0 + chol_factor) * (1.0 + bp_factor) * smoke_factor)
        .min(30.0);

    let interp = match risk {
        r if r < 1.0 => "Низкий риск (<1%)",
        r if r < 5.0 => "Умеренный риск (1-4%)",
        r if r < 10.0 => "Высокий риск (5-9%)",
        _ => "Очень высокий риск (>=10%)",
    };

    Ok(Json(CalculatorResult {
        result: (risk * 10.0).round() / 10.0,
        unit: "%".into(),
        interpretation: interp.into(),
    }))
}
