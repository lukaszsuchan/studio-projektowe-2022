package trafficsimulation.restapi.model.request;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import trafficsimulation.restapi.model.SimulationParamsDto;

@Data
@AllArgsConstructor
@NoArgsConstructor
public class SimulationParamsRequest {
   private SimulationParamsDto simulationParams;
}
