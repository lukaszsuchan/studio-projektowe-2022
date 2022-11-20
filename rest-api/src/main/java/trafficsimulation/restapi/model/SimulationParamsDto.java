package trafficsimulation.restapi.model;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;

import java.time.DayOfWeek;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SimulationParamsDto {
    @NonNull
    private int hour;
    @NonNull
    private int minutes;
    @NonNull
    private DayOfWeek dayOfWeek;
    @NonNull
    private int heavyVehiclePercentage;
}
