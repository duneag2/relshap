SELECT
  p.application_id,

  ap.checking_status,
  ap.duration,
  ap.credit_history,
  ap.purpose,
  ap.credit_amount,
  ap.savings_status,
  ap.employment,
  ap.installment_commitment,
  p.personal_status,
  p.other_parties,
  ap.residence_since,
  ap.property_magnitude,
  p.age,
  ap.other_payment_plans,
  p.housing,
  ap.existing_credits,
  p.job,
  p.num_dependents,
  p.own_telephone,
  p.foreign_worker
FROM Application ap
JOIN Applicant p
  ON p.applicant_id = ap.applicant_id
ORDER BY p.application_id;
