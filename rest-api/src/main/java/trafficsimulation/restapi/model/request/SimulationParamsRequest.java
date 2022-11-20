package trafficsimulation.restapi.model.request;

import lombok.AllArgsConstructor;
import lombok.Data;

import java.time.DayOfWeek;

@Data
@AllArgsConstructor
public class SimulationParamsRequest {
    private int hour;
    private int minutes;
    private DayOfWeek dayOfWeek;
    private int heavyVehiclePercentage;
}
