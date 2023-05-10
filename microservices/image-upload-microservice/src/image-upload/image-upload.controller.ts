import { Body, Controller, Post, UploadedFile, UseFilters, UseGuards, UseInterceptors } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import axios from 'axios';
import { MulterFile } from 'multer';
import { ExternalAuthGuard } from 'src/guards/external-auth.guard';
import { memoryStorage } from 'multer';
import { ClassifyImageDto } from 'src/dto/ClassifyImageDto';
import { multerConfig } from 'src/multer.config';
import { ClipService } from 'src/services/clip.service';
import { ResnetService } from 'src/services/resnet.service';
import { ImageClassificationExceptionFilter } from 'src/filters/image-classification-exception.filter';
import { ImageClassificationException } from 'src/filters/exceptions/image-classification.exception';
import { ImageUploadException } from 'src/filters/exceptions/image-upload.exception';
import { ImageUploadExceptionFilter } from 'src/filters/image-upload-exception.filter';
import { FlaggingMicroserviceException } from 'src/filters/exceptions/flagging-microservice.exception';
import { FlaggingMicroserviceExceptionFilter } from 'src/filters/flagging-microservice-exception.filter';
import * as sharp from 'sharp';

@Controller('images')
export class ImageUploadController {
  constructor(private readonly clipService: ClipService, private readonly resnetService: ResnetService) {}

  @UseGuards(ExternalAuthGuard)
  @Post('upload')
  @UseInterceptors(FileInterceptor('image', multerConfig))
  @UseFilters(ImageUploadExceptionFilter)
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
      throw new ImageUploadException('Failed to upload image');
    }
  }

  @UseGuards(ExternalAuthGuard)
  @Post('classify')
  @UseInterceptors(
    FileInterceptor('image', {
      storage: memoryStorage(),
      limits: {
        fileSize: 5 * 1024 * 1024, // 5MB (adjust the value as needed)
      },
    }),
  )
  @UseFilters(ImageClassificationExceptionFilter, ImageUploadExceptionFilter, FlaggingMicroserviceExceptionFilter)
  async classifyImage(@UploadedFile() image: MulterFile, @Body() body: ClassifyImageDto) {
    const classes = body.classes;
    const flaggingThreshold = body.flaggingThreshold;

    if (!image) {
      throw new ImageUploadException('Image data is missing');
    }

    const resizedImageBuffer = await sharp(image.buffer).resize({ width: 224, height: 224, fit: 'cover' }).webp({ quality: 80 }).toBuffer();

    image.buffer = resizedImageBuffer;

    let clipResult;
    let resnetResult;
    try {
      clipResult = await this.clipService.classifyImage(image, classes);
      resnetResult = await this.resnetService.classifyImage(image);
    } catch (error) {
      throw new ImageClassificationException('Classification Failed');
    }

    // Process the classification results
    const clipPrediction = clipResult.clipPrediction;
    const clipLabelProbabilities = clipResult.clipLabelProbabilities;
    const resnetPrediction = resnetResult.resnetPrediction;
    const resnetLabelProbabilities = resnetResult.resnetLabelProbabilities;

    // Flag the image if necessary

    /*  try {
      if (
        shouldFlagImage(
          clipLabelProbabilities,
          resnetLabelProbabilities,
          flaggingThreshold && +flaggingThreshold,
        )
      ) {
        await flagImage(image.originalname);
      }
    } catch (error) {
      throw new FlaggingMicroserviceException('Flagging Failed');
    }
 */
    // Return the classification results
    return {
      clipPrediction,
      clipLabelProbabilities,
      resnetPrediction,
      resnetLabelProbabilities,
    };
  }
}

function shouldFlagImage(clipLabelProbabilities, resnetLabelProbabilities, flaggingThreshold = 0.9): boolean {
  // Check if any label probability exceeds the threshold
  for (const label in clipLabelProbabilities) {
    if (clipLabelProbabilities[label] > flaggingThreshold || resnetLabelProbabilities[label] > flaggingThreshold) {
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
