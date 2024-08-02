use candle_core::{Device, Tensor};
use anyhow::Result;

pub fn pack_tensors(tensors: Vec<Tensor>, device: &Device) -> Result<Tensor> {
    let num_tensors = tensors.len();
    let mut prepared_tensors = Vec::from([
        Tensor::from_slice(&*[num_tensors], (1), device)?,
    ]);

    for tensor in tensors {
        let shape_info = tensor.shape().clone().into_dims();

        let shape_info_f64 = shape_info.clone().into_iter().map(|x| f64::from(x)).collect();

        let shape_info_len = shape_info.len();

        let flattened_tensor = tensor.flatten_all()?;

        prepared_tensors.push(Tensor::from_slice(&*[shape_info_len], 1, device)?);
        prepared_tensors.push(Tensor::from_vec(shape_info_f64, shape_info_len, device)?);
        prepared_tensors.push(flattened_tensor);
    }

    Ok(Tensor::cat(&prepared_tensors, 0)?)
}

pub fn unpack_tensors(tensor: Tensor) -> Result<Vec<Tensor>> {

    let mut unpacked_tensors: Vec<Tensor> = Vec::new();

    let num_tensors = tensor.get(0)?.to_scalar() as usize;

    let mut idx = 1;

    for i in 0..num_tensors {
        let shape_info_len = tensor.get(idx)?.to_scalar() as usize;
        idx += 1;

        let shape_info: Vec<usize> = tensor
            .narrow(0, idx, shape_info_len)?
            .to_vec1()?.into_iter().map(|x| usize::from(x)).collect();

        idx += shape_info_len;

        let num_elements = shape_info.iter().product();

        let extracted = tensor.narrow(0, idx, num_elements)?.reshape(shape_info)?;
        idx += num_elements;

        unpacked_tensors.push(extracted);
    }

    Ok(unpacked_tensors)
}
