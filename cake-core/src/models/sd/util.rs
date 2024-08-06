use candle_core::{Device, Tensor};
use anyhow::Result;
use candle_core::utils::{cuda_is_available, metal_is_available};
use candle_transformers::models::stable_diffusion::StableDiffusionConfig;
use crate::cake::Context;
use crate::StableDiffusionVersion;

pub fn pack_tensors(tensors: Vec<Tensor>, device: &Device) -> Result<Tensor> {
    let num_tensors = tensors.len();
    let mut prepared_tensors = Vec::from([
        Tensor::from_slice(&[num_tensors as i64], 1, device)?,
    ]);

    for tensor in tensors {
        let shape_info = tensor.shape().clone().into_dims();

        let shape_info_i64 = shape_info.clone().into_iter().map(|x| x as i64).collect();

        let shape_info_len = shape_info.len();

        let flattened_tensor = tensor.flatten_all()?;

        prepared_tensors.push(Tensor::from_slice(&[shape_info_len as i64], 1, device)?);
        prepared_tensors.push(Tensor::from_vec(shape_info_i64, shape_info_len, device)?);
        prepared_tensors.push(flattened_tensor);
    }

    Ok(Tensor::cat(&prepared_tensors, 0)?)
}

pub fn unpack_tensors(tensor: &Tensor) -> Result<Vec<Tensor>> {

    let mut unpacked_tensors: Vec<Tensor> = Vec::new();

    let num_tensors: i64 = tensor.get(0)?.to_scalar()?;

    let mut idx:i64 = 1;

    for _i in 0..num_tensors {
        let shape_info_len: i64 = tensor.get(idx as usize)?.to_scalar()?;
        idx += 1;

        let shape_info: Vec<i64> = tensor
            .narrow(0, idx as usize, shape_info_len as usize)?
            .to_vec1()?.into_iter().collect();

        idx += shape_info_len;

        let num_elements: i64 = shape_info.iter().product();

        let shape_info_usize: Vec<_> = shape_info.iter().map(|&x|{x as usize}).collect();

        let extracted = tensor.narrow(0, idx as usize, num_elements as usize)?.reshape(shape_info_usize)?;
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
    let width= ctx.args.sd_args.width;
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
