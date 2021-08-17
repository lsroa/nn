use crate::matrix::{Matrix, MatrixOperation};

#[derive(Debug, Copy, Clone)]
struct ActivationFunction {
    func: fn(i16) -> i16,
    d_func: fn(i16) -> i16,
}

impl ActivationFunction {
    fn new(a: fn(i16) -> i16, b: fn(i16) -> i16) -> Self {
        ActivationFunction { func: a, d_func: b }
    }
}

static SIGMOID: ActivationFunction = ActivationFunction {
    func: |x| 1 as i16 / (1 as i16 + -x),
    d_func: |y| y * (1 as i16 - y),
};

static TANH: ActivationFunction = ActivationFunction {
    func: |x| x * x,
    d_func: |y| 1 as i16 - (y * y),
};

pub struct NeuronalNetwork {
    activation_func: ActivationFunction,
    learning_rate: f32,
    weights_ih: Matrix,
    weights_ho: Matrix,
    bias_h: Matrix,
    bias_o: Matrix,
    input_nodes: usize,
    output_nodes: usize,
    hidden_nodes: usize,
}

impl NeuronalNetwork {
    fn new(input_nodes: usize, output_nodes: usize, hidden_nodes: usize) -> Self {
        let weights_ho = Matrix::new(output_nodes, hidden_nodes).randomize().copy();
        let weights_ih = Matrix::new(hidden_nodes, input_nodes).randomize().copy();

        let bias_h = Matrix::new(hidden_nodes, 1).randomize().copy();
        let bias_o = Matrix::new(output_nodes, 1).randomize().copy();

        NeuronalNetwork {
            activation_func: SIGMOID,
            learning_rate: 0.1,
            input_nodes,
            output_nodes,
            hidden_nodes,
            weights_ho,
            weights_ih,
            bias_h,
            bias_o,
        }
    }

    fn copy(a: &Self) -> Self {
        let NeuronalNetwork {
            input_nodes,
            output_nodes,
            hidden_nodes,
            activation_func,
            learning_rate,
            ..
        } = *a;

        NeuronalNetwork {
            output_nodes,
            input_nodes,
            hidden_nodes,
            activation_func,
            learning_rate,
            weights_ih: a.weights_ih.copy(),
            weights_ho: a.weights_ho.copy(),
            bias_h: a.bias_h.copy(),
            bias_o: a.bias_o.copy(),
        }
    }
}

impl NeuronalNetwork {
    pub fn set_activation_func(&mut self, func: Option<ActivationFunction>) {
        self.activation_func = match func {
            None => SIGMOID,
            Some(func) => func,
        }
    }

    pub fn set_learning_rate(&mut self, learning_rate: Option<f32>) {
        self.learning_rate = match learning_rate {
            None => 0.1,
            Some(x) => x,
        }
    }

    pub fn predict(&mut self, input_array: &[i16]) -> Vec<i16> {
        let inputs = Matrix::from_array(input_array);
        let mut hidden = Matrix::product(&self.weights_ih, &inputs);

        hidden.add(self.bias_h.copy());
        hidden.map(self.activation_func.func);

        let mut output = Matrix::product(&self.weights_ho, &hidden);
        output.add(self.bias_o.copy());

        output.to_array()
    }

    pub fn train(&mut self, input_array: &[i16], target_array: &[i16]) {
        let input = Matrix::from_array(input_array);
        let mut hidden = Matrix::product(&self.weights_ih, &input);

        hidden.add(self.bias_o.copy());
        hidden.map(self.activation_func.func);

        let outputs = Matrix::product(&self.weights_ho, &hidden);

        outputs.add(self.bias_h.copy());
        outputs.map(self.activation_func.func);

        let target = Matrix::from_array(target_array);

        let output_errors = Matrix::new(4, 4);

        let mut gradients = Matrix::static_map(outputs, self.activation_func.d_func);
        gradients
            .multiply(output_errors)
            .multiply(self.learning_rate);
    }

    pub fn mutate(&mut self, func: fn(i16) -> i16) {
        self.weights_ih.map(func);
        self.weights_ho.map(func);
        self.bias_h.map(func);
        self.bias_o.map(func);
    }
}
