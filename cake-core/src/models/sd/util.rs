use crate::cake::Context;
use crate::StableDiffusionVersion;
use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_core::{Device, Tensor};
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;

pub fn pack_tensors(tensors: Vec<Tensor>, device: &Device) -> Result<Tensor> {
    let num_tensors = tensors.len();
    let mut prepared_tensors = Vec::from([Tensor::from_slice(&[num_tensors as f32], 1, device)?]);

    for tensor in tensors {
        let shape_info = tensor.shape().clone().into_dims();

        let shape_info_f32 = shape_info.clone().into_iter().map(|x| x as f32).collect();

        let shape_info_len = shape_info.len();

        let flattened_tensor = tensor.flatten_all()?.to_dtype(candle_core::DType::F32)?;

        prepared_tensors.push(Tensor::from_slice(&[shape_info_len as f32], 1, device)?);
        prepared_tensors.push(Tensor::from_vec(shape_info_f32, shape_info_len, device)?);
        prepared_tensors.push(flattened_tensor);
    }

    Ok(Tensor::cat(&prepared_tensors, 0)?)
}

pub fn unpack_tensors(tensor: &Tensor) -> Result<Vec<Tensor>> {
    let mut unpacked_tensors: Vec<Tensor> = Vec::new();

    let num_tensors: f32 = tensor.get(0)?.to_scalar()?;
    let num_tensors_i32 = num_tensors as i32;

    let mut idx: i32 = 1;

    for _i in 0..num_tensors_i32 {
        let shape_info_len: f32 = tensor.get(idx as usize)?.to_scalar()?;

        idx += 1;

        let shape_info: Vec<i32> = tensor
            .narrow(0, idx as usize, shape_info_len as usize)?
            .to_vec1()?
            .into_iter()
            .map(|x: f32| x as i32)
            .collect();

        idx += shape_info_len as i32;

        let num_elements: i32 = shape_info.iter().product();

        let shape_info_usize: Vec<_> = shape_info.iter().map(|&x| x as usize).collect();

        let extracted = tensor
            .narrow(0, idx as usize, num_elements as usize)?
            .reshape(shape_info_usize)?;
        idx += num_elements;

        unpacked_tensors.push(extracted);
    }

    Ok(unpacked_tensors)
}

pub fn get_device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub fn get_sd_config(ctx: &Context) -> Result<StableDiffusionConfig> {
    let height = ctx.args.sd_args.height;
    let width = ctx.args.sd_args.width;
    let sliced_attention_size = ctx.args.sd_args.sliced_attention_size;
    let sd_config = match ctx.args.sd_args.sd_version {
        StableDiffusionVersion::V1_5 => {
            StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::V2_1 => {
            StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::Xl => {
            StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
        }
        StableDiffusionVersion::Turbo => StableDiffusionConfig::sdxl_turbo(
            ctx.args.sd_args.sliced_attention_size,
            ctx.args.sd_args.height,
            ctx.args.sd_args.width,
        ),
    };
    Ok(sd_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_pack_unpack_single_tensor() {
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3), &device).unwrap();
        let packed = pack_tensors(vec![t.clone()], &device).unwrap();
        let unpacked = unpack_tensors(&packed).unwrap();

        assert_eq!(unpacked.len(), 1);
        assert_eq!(unpacked[0].shape().dims(), &[2, 3]);
        let orig: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        let recv: Vec<f32> = unpacked[0].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, recv);
    }

    #[test]
    fn test_pack_unpack_multiple_tensors() {
        let device = Device::Cpu;
        let t1 = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (1, 3), &device).unwrap();
        let t2 = Tensor::from_vec(vec![4.0f32, 5.0, 6.0, 7.0], (2, 2), &device).unwrap();
        let t3 = Tensor::from_vec(vec![8.0f32], (1, 1), &device).unwrap();

        let packed = pack_tensors(vec![t1.clone(), t2.clone(), t3.clone()], &device).unwrap();
        let unpacked = unpack_tensors(&packed).unwrap();

        assert_eq!(unpacked.len(), 3);
        assert_eq!(unpacked[0].shape().dims(), &[1, 3]);
        assert_eq!(unpacked[1].shape().dims(), &[2, 2]);
        assert_eq!(unpacked[2].shape().dims(), &[1, 1]);

        let v0: Vec<f32> = unpacked[0].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v0, vec![1.0, 2.0, 3.0]);
        let v1: Vec<f32> = unpacked[1].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v1, vec![4.0, 5.0, 6.0, 7.0]);
        let v2: Vec<f32> = unpacked[2].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(v2, vec![8.0]);
    }

    #[test]
    fn test_pack_unpack_3d_tensor() {
        let device = Device::Cpu;
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_vec(data, (2, 3, 4), &device).unwrap();

        let packed = pack_tensors(vec![t.clone()], &device).unwrap();
        let unpacked = unpack_tensors(&packed).unwrap();

        assert_eq!(unpacked.len(), 1);
        assert_eq!(unpacked[0].shape().dims(), &[2, 3, 4]);
        let orig: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
        let recv: Vec<f32> = unpacked[0].flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, recv);
    }

    #[test]
    fn test_get_device_cpu_flag_returns_cpu() {
        let dev = get_device(true).unwrap();
        assert!(matches!(dev, Device::Cpu));
    }

    #[test]
    fn test_pack_unpack_f16_cast_to_f32() {
        // pack_tensors casts to F32 internally, so F16 input data is preserved
        // only approximately (within F16 precision)
        let device = Device::Cpu;
        let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], (3,), &device)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let packed = pack_tensors(vec![t.clone()], &device).unwrap();
        let unpacked = unpack_tensors(&packed).unwrap();

        assert_eq!(unpacked.len(), 1);
        assert_eq!(unpacked[0].shape().dims(), &[3]);
        // Unpacked is F32 (pack converts to F32)
        let recv: Vec<f32> = unpacked[0].to_vec1().unwrap();
        assert_eq!(recv, vec![1.0, 2.0, 3.0]);
    }
}
