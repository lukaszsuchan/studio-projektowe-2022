package trafficsimulation.restapi.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.io.File;
import java.io.IOException;

@Service
@RequiredArgsConstructor
public class ObjectToJsonFile {

    private static final String PATH = "src/main/resources/simulation-params.json";

    private final ObjectMapper objectMapper;

    public void parseObjectToJsonFile(Object object) {
        File file = new File(PATH);

        try {
            objectMapper.writeValue(file, object);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
