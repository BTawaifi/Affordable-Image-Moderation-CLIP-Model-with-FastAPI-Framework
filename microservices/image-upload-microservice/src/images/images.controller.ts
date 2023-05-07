import {
  Body,
  Controller,
  Post,
  UploadedFile,
  UseGuards,
  UseInterceptors,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import axios from 'axios';
import { diskStorage } from 'multer';
import { extname } from 'path';
import { MulterFile } from 'multer';
import { ExternalAuthGuard } from 'src/guards/external-auth.guard';
import * as FormData from 'form-data';
import { memoryStorage } from 'multer';
import { ClassifyImageDto } from 'src/dto/ClassifyImageDto';
import { multerConfig } from 'src/multer.config';

@Controller('images')
export class ImagesController {
  @UseGuards(ExternalAuthGuard)
  @Post('upload')
  @UseInterceptors(FileInterceptor('image', multerConfig))
  async uploadImage(@UploadedFile() image: MulterFile) {
    try {
      // Return a response with the uploaded image information
      return {
        message: 'Image uploaded successfully',
        filename: image.filename,
        size: image.size,
      };
    } catch (error) {
      // Handle any errors that occur during image upload
      throw new Error('Failed to upload image');
    }
  }

  //@UseGuards(ExternalAuthGuard)
  @Post('classify')
  @UseInterceptors(
    FileInterceptor('image', {
      storage: memoryStorage(),
      limits: {
        fileSize: 5 * 1024 * 1024, // 5MB (adjust the value as needed)
      },
    }),
  )
  async classifyImage(
    @UploadedFile() image: MulterFile,
    @Body() body: ClassifyImageDto,
  ) {
    try {
      const classes = body.classes;

      if (!image) {
        throw new Error('Image data is missing');
      }

      // Create FormData objects for each microservice request
      const clipFormData = new FormData();
      //const resnetFormData = new FormData();

      // Append the image to FormData objects
      clipFormData.append('image', image.buffer, {
        filename: image.originalname,
        contentType: image.mimetype,
      });

      classes && clipFormData.append('classes', classes);

      /*resnetFormData.append('image', image.buffer, {
        filename: image.originalname,
        contentType: image.mimetype,
      }); */

      // Upload the image to the CLIP microservice for classification
      const clipResult = await axios.post(
        'http://localhost:3000/clip/classify',
        clipFormData,
        {
          headers: clipFormData.getHeaders(),
        },
      );

      /*// Upload the image to the ResNet microservice for classification
      const resnetResult = await axios.post(
        'http://localhost:4000/resnet/classify',
        resnetFormData,
        {
          headers: resnetFormData.getHeaders(),
        },
      ); */

      // Process the classification results
      const clipPrediction = clipResult.data.prediction;
      const clipLabelProbabilities = clipResult.data.label_probabilities;
      /*       const resnetPrediction = resnetResult.data.prediction;
      const resnetLabelProbabilities = resnetResult.data.labelProbabilities; */

      // Flag the image if necessary
      /*       if (shouldFlagImage(clipLabelProbabilities, resnetLabelProbabilities)) {
        await flagImage(image.originalname);
      } */

      // Return the classification results
      return {
        clipPrediction,
        clipLabelProbabilities,
        /*resnetPrediction,
        resnetLabelProbabilities, */
      };
    } catch (error) {
      // Handle any errors that occur during image classification
      console.log(error);

      throw new Error('Failed to classify image');
    }
  }
}

function shouldFlagImage(
  clipLabelProbabilities: { [key: string]: number },
  resnetLabelProbabilities: { [key: string]: number },
): boolean {
  // Define the threshold for flagging an image
  const flaggingThreshold = 0.9;

  // Check if any label probability exceeds the threshold
  for (const label in clipLabelProbabilities) {
    if (
      clipLabelProbabilities[label] > flaggingThreshold ||
      resnetLabelProbabilities[label] > flaggingThreshold
    ) {
      return true;
    }
  }

  return false;
}

async function flagImage(filename: string): Promise<void> {
  // Send a request to the flagging-microservice to store the flagged image in the database
  await axios.post('http://localhost:6000/flagging/images', { filename });

  console.log(`Image ${filename} flagged for manual review`);
}
