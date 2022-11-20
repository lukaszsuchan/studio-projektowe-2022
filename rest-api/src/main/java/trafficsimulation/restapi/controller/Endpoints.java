package trafficsimulation.restapi.controller;

import lombok.NonNull;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import trafficsimulation.restapi.model.request.SimulationParamsRequest;

import javax.validation.Valid;

import static trafficsimulation.restapi.controller.Endpoints.CONTROLLER_PATH;

@RequestMapping(CONTROLLER_PATH)
interface Endpoints {

    String CONTROLLER_PATH = "/rest-api";

    @ResponseStatus(HttpStatus.OK)
    @PostMapping(value = "/simulation", consumes = MediaType.APPLICATION_JSON_VALUE)
    ResponseEntity<Void> setSimulationParams(@Valid @NonNull @RequestBody SimulationParamsRequest request);
}
