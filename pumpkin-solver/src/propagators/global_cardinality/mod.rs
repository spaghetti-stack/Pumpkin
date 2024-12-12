pub(crate) mod gcc_lower_upper_2;
pub(crate) mod simple_gcc_lower_upper;

#[derive(Clone, Debug, Copy)]
pub struct Values {
    pub value: i32,
    pub omin: i32,
    pub omax: i32,
}
