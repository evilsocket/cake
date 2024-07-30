use crate::cake::Context;
use crate::models::{Generator, ImageGenerationParameters, ImageGenerator};

pub struct SD15 {

}

impl Generator for SD15 {
    type Shardable = ();
    const MODEL_NAME: &'static str = "SD15";

    async fn load(context: Context) -> anyhow::Result<Option<Box<Self>>> {
        todo!()
    }
}

impl ImageGenerator for SD15 {
    async fn generate_image(&mut self, params: &ImageGenerationParameters) -> anyhow::Result<String> {
        todo!()
    }
}