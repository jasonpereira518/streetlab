/** The six-widget telemetry strip along the bottom of the app. */
import { LanePosition } from './telemetry/LanePosition';
import { RadarPolar } from './telemetry/RadarPolar';
import { Speedometer } from './telemetry/Speedometer';
import { SteeringWheel } from './telemetry/SteeringWheel';
import { TrajectoryGraph } from './telemetry/TrajectoryGraph';
import { VehicleStatus } from './telemetry/VehicleStatus';

export function TelemetryRow() {
  return (
    <section className="telemetry" aria-label="Telemetry">
      <Speedometer />
      <LanePosition />
      <RadarPolar />
      <VehicleStatus />
      <TrajectoryGraph />
      <SteeringWheel />
    </section>
  );
}
