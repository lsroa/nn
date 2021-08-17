extern crate rand;

use rand::thread_rng;
use rand::Rng;
#[derive(Debug, Clone)]
pub struct Matrix {
    cols: usize,
    rows: usize,
    data: Vec<Vec<i16>>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            cols,
            rows,
            data: vec![vec![0; cols]; rows],
        }
    }

    pub fn from_array(list: &[i16]) -> Self {
        Matrix::new(list.len(), 1).map_with_location(|_, i, _| list[i])
    }

    pub fn static_map(m: Self, func: impl Fn(i16, usize, usize) -> i16) -> Self {
        Matrix::new(m.cols, m.rows).map_with_location(|e, i, j| func(e, i, j))
    }

    pub fn transpose(m: &mut Self) -> Self {
        let copy = m.clone().data;
        m.map_with_location(|_x, i, j| copy[j][i])
    }

    pub fn product(a: &Self, b: &Self) -> Self {
        Matrix::new(b.cols, a.rows).map_with_location(|_, i, j| {
            let mut sum = 0;
            for k in 0..a.cols {
                sum += a.data[i][k] * b.data[k][j];
            }
            sum
        })
    }
}

impl Matrix {
    pub fn copy(&self) -> Matrix {
        Matrix::new(self.rows, self.cols).map_with_location(|_, i, j| self.data[i][j])
    }

    pub fn randomize(&mut self) -> &mut Self {
        let mut rng = thread_rng();
        self.map(|_| rng.gen_range(-10, 10))
    }

    pub fn to_array(&mut self) -> Vec<i16> {
        let mut arr = vec![];
        self.map(|value| {
            arr.push(value);
            value
        });
        arr
    }

    pub fn map(&mut self, mut func: impl FnMut(i16) -> i16) -> &mut Self {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.data[i][j];
                self.data[i][j] = func(value);
            }
        }
        self
    }

    pub fn map_with_location(&mut self, mut func: impl FnMut(i16, usize, usize) -> i16) -> Self {
        let mut new = Matrix::new(self.rows, self.cols);
        for i in 0..new.rows {
            for j in 0..new.cols {
                let value = self.data[i][j];
                new.data[i][j] = func(value, i, j);
            }
        }
        new
    }

    pub fn print(&self) {
        for i in &self.data {
            for j in i {
                print!(" {}  ", j);
            }
            println!();
        }
        println!();
    }
}

pub trait MatrixOperation<T> {
    fn add(&mut self, t: T) -> &mut Matrix;
    fn multiply(&mut self, t: T) -> &mut Matrix;
}

impl MatrixOperation<i16> for Matrix {
    fn add(&mut self, n: i16) -> &mut Self {
        self.map(|e| e + n)
    }

    fn multiply(&mut self, n: i16) -> &mut Matrix {
        self.map(|e| e * n)
    }
}

impl MatrixOperation<f32> for Matrix {
    fn add(&mut self, n: f32) -> &mut Self {
        self.map(|e| e as f32 + n)
    }

    fn multiply(&mut self, n: f32) -> &mut Matrix {
        self.map(|e| e as f32 * n)
    }
}

impl MatrixOperation<Matrix> for Matrix {
    fn add(&mut self, m: Matrix) -> &mut Self {
        todo!()
    }

    fn multiply(&mut self, t: Matrix) -> &mut Matrix {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initializer() {
        let m = Matrix::new(5, 4);
        assert_eq!(m.rows, 5);
        assert_eq!(m.cols, 4);
    }

    #[test]
    fn product() {
        let mut a = Matrix::new(2, 3);
        a.data[0] = vec![1, 2, 3];
        a.data[1] = vec![4, 5, 6];

        let mut b = Matrix::new(3, 2);
        b.data[0] = vec![7, 8];
        b.data[1] = vec![9, 10];
        b.data[2] = vec![11, 12];

        let ab = Matrix::product(&a, &b);

        assert_eq!(ab.data, vec![vec![58, 64], vec![139, 154]]);
    }
    #[test]
    fn from_array() {
        let m = Matrix::from_array(&[1, 2, 3, 4]);

        assert_eq!(m.data, vec![vec![1], vec![2], vec![3], vec![4]]);
    }

    #[test]
    fn to_array() {
        let mut m = Matrix::new(2, 2);
        m.data[0] = vec![1, 2];
        m.data[1] = vec![3, 4];

        let arr = m.to_array();

        assert_eq!(arr, vec![1, 2, 3, 4]);
    }
    #[test]
    fn copy() {
        let mut m = Matrix::new(3, 3);
        m.randomize();

        let c = m.copy();

        assert_eq!(c.data, m.data);
        assert_eq!(c.rows, m.rows);
        assert_eq!(c.cols, m.cols);
    }
}
