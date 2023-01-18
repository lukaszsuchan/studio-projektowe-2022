import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment';

@Component({
  selector: 'app-form',
  templateUrl: './form.component.html',
  styleUrls: ['./form.component.css']
})

export class FormComponent implements OnInit {

  loading = false;
  pedestrianLevel = 3
  trafficLevel = 30
  simulationUrl = "http://localhost:8000"

  constructor(private fb: FormBuilder, private http: HttpClient) { }

  ngOnInit(): void {
  }

  SimulationForm = this.fb.group({
    vmax: ['', [Validators.required, Validators.pattern("^([1-9]|10)$")]],
    heavyVehiclesPercentage: ['', [Validators.required, Validators.pattern("([0-9]|[1-9][0-9]|100)")]]
  })

  public onSubmit(): void{
    console.log()
    if (this.SimulationForm.valid){ 
      let busPerc: number = +this.SimulationForm.get('heavyVehiclesPercentage')?.value;
      busPerc =  Math.round(busPerc / 100 * this.trafficLevel);
      const simulationParams = {
        Vmax: this.SimulationForm.get('vmax')?.value,
        BusesPercentage: busPerc,
        PedestrianLevel: this.pedestrianLevel,
        TrafficLevel: this.trafficLevel
      };
      this.http.post<any>("https://localhost:7090/Simulation/run", simulationParams).subscribe(res => {this.loading = true;
      setTimeout(() => {
        window.location.href = this.simulationUrl;
    }, 10000);});
    } else {
      this.markAllAsChecked();
    }
  }

  private markAllAsChecked(): void{
    Object.keys(this.SimulationForm.controls).forEach(key => {
      this.SimulationForm.controls[key].markAsTouched();
    });
  }

  onPedestrianLevelChange(e: any){
    if(e.target.value === "small"){
      this.pedestrianLevel = 1
    } else if(e.target.value === "medium"){
      this.pedestrianLevel = 3
    } else {
      this.pedestrianLevel = 5
    }
    console.log(this.pedestrianLevel)
  }

  onTrafficLevelChange(e: any){
    if(e.target.value === "small"){
      this.trafficLevel = 10
    } else if(e.target.value === "medium"){
      this.trafficLevel = 25
    } else {
      this.trafficLevel = 35
    }
    console.log(this.trafficLevel);
  }

}
