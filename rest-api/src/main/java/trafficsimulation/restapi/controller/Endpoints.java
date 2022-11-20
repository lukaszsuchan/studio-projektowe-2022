package trafficsimulation.restapi.controller;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import trafficsimulation.restapi.model.request.SimulationParamsRequest;

import static trafficsimulation.restapi.controller.Endpoints.CONTROLLER_PATH;

@RequestMapping(CONTROLLER_PATH)
interface Endpoints {

    String CONTROLLER_PATH = "/rest-api";

    @ResponseStatus(HttpStatus.OK)
    @PostMapping( "/simulation")
    ResponseEntity<Void> setSimulationParams(@RequestBody SimulationParamsRequest request);
}
