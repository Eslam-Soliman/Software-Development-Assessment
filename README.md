Software-Development-Assessment
==============


## Description
This is a coding assessment for [Genify.ai](https://www.genify.ai/). 

This is a basic API built on top of model generated by this [Kaggle notebook](https://www.kaggle.com/sudalairajkumar/when-less-is-more)

## How to run
- Clone the repo
- Navigate to 'src' folder
- Run 'python .\run.py'
- The endpoint should be up on http://127.0.0.1:5000

## Example query
http://127.0.0.1:5000?fecha_dato=2016-06-28&ncodpers=15889&ind_empleado=F&pais_residencia=ES&sexo=V&age=56&fecha_alta=1995-01-16&antiguedad=256&tiprel_1mes=A&cod_prov=28&ind_actividad_cliente=1&renta=326124.90&segmento=01%20-%20TOP

## API Documentation
- The API has a single POST endpoint for recommending banking products to users based on their banking and social information
- The API POST endpoint only accepts parameters send in the query URL

- Endpoint parameters

|       Parameters    | Is required |
|---------------------|-------------|
|fecha_dato           |Yes          |
|ncodpers             |Yes          |
|ind_empleado         |No           |
|pais_residencia      |No           |
|ind_empleado         |No           |
|sexo                 |No           |
|age                  |No           |
|fecha_alta           |No           |
|antiguedad           |No           |
|tiprel_1mes          |No           |
|cod_prov             |No           |
|ind_actividad_cliente|No           |     
|renta                |No           |
|segmento             |No           |