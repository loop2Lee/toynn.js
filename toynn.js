class ActivationFunction {
  constructor(func, dfunc) {
    this.func = func;
    this.dfunc = dfunc;
  }
}

let sigmoid = new ActivationFunction(
  x => 1 / (1 + Math.exp(-x)),
  y => y * (1 - y)
);

let tanh = new ActivationFunction(
  x => Math.tanh(x),
  y => 1 - (y * y)
);

class toynn{
  constructor(input_nodes, hidden_layers, output_nodes, learning_rate, activation_function){
    //setups
    this.input_nodes = input_nodes;
    this.hidden_layers = hidden_layers;
    this.output_nodes = output_nodes;
    //create matrixs
    this.weight_matrixs = [];
    this.bias_matrixs = [];
    var start = input_nodes;
    var holder;
    for(var i=0;i<hidden_layers.length;i++){
      //loop through the hidden layers
      this.weight_matrixs.push(new Matrix(hidden_layers[i],start).randomize());
      this.bias_matrixs.push(new Matrix(hidden_layers[i],1).randomize());
      start = hidden_layers[i];
    }
    //matrix between hidden and output
    this.weight_matrixs.push(new Matrix(output_nodes,start).randomize());
    this.bias_matrixs.push(new Matrix(output_nodes,1).randomize());
    this.learning_rate = learning_rate;
    if(activation_function=='sigmoid'){
      this.activation_function = sigmoid;
    }else if(activation_function=='tanh'){
      this.activation_function = tanh;
    }
  }

  train(input_array, target_array){
    //Feed Forward, Generate Output
    var previous = Matrix.fromArray(input_array);
    var layer_record = [Matrix.transpose(previous)];
    for (var i=0;i<this.weight_matrixs.length;i++) {
      previous = Matrix.multiply(this.weight_matrixs[i],previous);
      previous.add(this.bias_matrixs[i]);
      previous.map(this.activation_function.func);
      layer_record.push(Matrix.transpose(previous));  //last of record is the output
    }

    //Backpropergation, Calculate Error
    layer_record.reverse();
    this.weight_matrixs.reverse();
    this.bias_matrixs.reverse();
    var targets = Matrix.fromArray(target_array);
    var errors = Matrix.transpose(Matrix.subtract(Matrix.transpose(targets),layer_record[0]));
    var error_holder;
    var gradients; //learning_rate*error
    for(var i=0;i<this.weight_matrixs.length;i++){
      error_holder = Matrix.multiply(Matrix.transpose(this.weight_matrixs[i]),errors);  //Calcuate hidden_errors
      gradients = Matrix.transpose(Matrix.map(layer_record[i],this.activation_function.dfunc));
      gradients.multiply(errors);
      gradients.multiply(this.learning_rate);
      this.weight_matrixs[i].add(Matrix.multiply(gradients,layer_record[i+1]));
      this.bias_matrixs[i].add(gradients);
      errors = error_holder;
    }
    this.weight_matrixs.reverse();
    this.bias_matrixs.reverse();
    //Calculate Deltas
  }

  predict(input_array) {
    //Feed Forward, Generate Output
    var previous = Matrix.fromArray(input_array);
    for (var i=0;i<this.weight_matrixs.length;i++) {
      previous = Matrix.multiply(this.weight_matrixs[i],previous);
      previous.add(this.bias_matrixs[i]);
      previous.map(this.activation_function.func);
    }
    // Sending back to the caller!
    return previous.toArray();
  }
}
