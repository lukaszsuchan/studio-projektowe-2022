import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormControl, FormGroup, Validators, ReactiveFormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { environment } from 'src/environments/environment';

@Component({
  selector: 'app-form',
  templateUrl: './form.component.html',
  styleUrls: ['./form.component.css']
})
export class FormComponent implements OnInit {

  loading = false;
  simulationUrl = "http://localhost:8000"

  constructor(private fb: FormBuilder, private http: HttpClient) { }

  ngOnInit(): void {
  }

  SimulationForm = this.fb.group({
    hour: ['', [Validators.required, Validators.pattern("^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$")]],
    weekday: ['', Validators.required],
    heavyVehiclesPercentage: ['', [Validators.required, Validators.pattern("([0-9]|[1-9][0-9]|100)")]]
  })

  public onSubmit(): void{
    if (this.SimulationForm.valid){
      const ParamsDto = {
        hour: this.SimulationForm.get('hour')?.value,
        weekday: this.SimulationForm.get('weekday')?.value,
        heavyVehiclesPercentage: this.SimulationForm.get('heavyVehiclesPercentage')?.value
      };
      this.http.post<any>(environment.urlApi + "/setParams", ParamsDto).subscribe(response => console.log(response));
      this.loading = true;
      setTimeout(() => {
        window.location.href = this.simulationUrl;
    }, 5000);
    } else {
      this.markAllAsChecked();
    }
  }

  private markAllAsChecked(): void{
    Object.keys(this.SimulationForm.controls).forEach(key => {
      this.SimulationForm.controls[key].markAsTouched();
    });
  }

}
