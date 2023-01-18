using backend.Entities;

namespace backend.Services
{
    public class SimulationService
    {
        public static void run(SimulationParams simulationParams)
        {
            setParams(simulationParams);
            System.Threading.Thread.Sleep(1000);
            System.Diagnostics.Process.Start("run_sim.bat");
        }

        public static void setParams(SimulationParams simulationParams)
        {
            StreamWriter sw = new StreamWriter(@"../../middleware/params.txt");
            sw.Write(simulationParams.Vmax.ToString() + "\n");
            sw.WriteLine(simulationParams.BusesPercentage.ToString());
            sw.WriteLine(simulationParams.PedestrianLevel.ToString());
            sw.WriteLine(simulationParams.TrafficLevel.ToString());
            sw.Close();
        }
    }
}
