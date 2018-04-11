toynn.js
-------------
An simple **JavaScript** implementation of a basic Neural Network.

Basic Usage
-------------

First, import **matrix.js** and **toynn.js** with the following HTML tags
```html
  <script src="matrix.js"></script>
  <script src="toynn.js"></script>
```

Then, initialize a Neural Network with:
  - four inputs
  - two hidden layers layers (both have three nodes)
  - one outputs
  - learning rate sets to 1
  - using sigmoid (now supports sigmoid and tanh) as activation function
```JavaScript
  new toynn(4, [3,3], 1, 1, 'sigmoid');
```

Third, train your data.
```JavaScript
  for (var j=0;j<30;j++){
    //train for 30 times
    for(var i=0;i<train_data_input.length;i++){
      nn.train(train_data_input[i],train_data_target[i]);	//train
    }
  }
```

Last, predict
```JavaScript
  nn.predict(test_data_input[i]); //returns predict output
```

Examples
-------------
- Iris (only this for now)

From Me
-------------
Try it yourself and have fun.
Zimo Xiao, inspired by The Coding Train.
