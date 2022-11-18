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

  constructor(private fb: FormBuilder, private http: HttpClient) { }

  ngOnInit(): void {
  }

  SimulationForm = this.fb.group({
    hour: ['', [Validators.required, Validators.pattern("^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$")]],
    weekday: ['', Validators.required],
    heavyVehiclesPercentage: ['', [Validators.required, Validators.pattern("([0-9]|[1-9][0-9]|100)")]]
  })

  public onSubmit(){
    if (this.SimulationForm.valid){
      const dto = {
        hour: this.SimulationForm.get('hour')?.value,
        weekday: this.SimulationForm.get('weekday')?.value,
        heavyVehiclesPercentage: this.SimulationForm.get('heavyVehiclesPercentage')?.value
      };
      console.log(dto);
      this.http.post<any>(environment.urlApi, dto).subscribe(response => console.log(response));
    }
  }

}
