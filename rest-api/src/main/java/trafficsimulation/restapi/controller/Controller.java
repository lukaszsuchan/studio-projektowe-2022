package trafficsimulation.restapi.controller;

import lombok.AllArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.RestController;
import trafficsimulation.restapi.model.request.SimulationParamsRequest;
import trafficsimulation.restapi.service.ObjectToJsonFile;


@RestController
@AllArgsConstructor
class Controller implements Endpoints{

    private final ObjectToJsonFile objectToJsonFile;

    @Override
    public ResponseEntity<Void> setSimulationParams(SimulationParamsRequest request) {
        objectToJsonFile.parseObjectToJsonFile(request);
        return ResponseEntity.ok().build();
    }
}
