using backend.Entities;
using backend.Services;
using Microsoft.AspNetCore.Mvc;

namespace backend.Controllers
{
    [ApiController]
    [Route("[controller]")]
    public class SimulationController : ControllerBase
    {
        [HttpPost("run")]
        public void StartSimulation([FromBody] SimulationParams simulationParams)
        {
            SimulationService.run(simulationParams);
        }
    }
}