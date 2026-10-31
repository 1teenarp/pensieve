CREATE TABLE taxi_zones (
  location_id  INT PRIMARY KEY,
  borough      TEXT,
  zone         TEXT,
  service_zone TEXT,
  updated_at   TIMESTAMPTZ DEFAULT now()
);
