import { FileInterceptor } from '@nestjs/platform-express';
import { ExternalAuthGuard } from '../guards/external-auth.guard';
import { ClassifyImageDto } from '../dto/ClassifyImageDto';
import { ImageClassificationExceptionFilter } from '../filters/image-classification-exception.filter';
import { ImageClassificationException } from '../filters/exceptions/image-classification.exception';
import { ImageUploadException } from '../filters/exceptions/image-upload.exception';
import { ImageUploadExceptionFilter } from '../filters/image-upload-exception.filter';
import * as sharp from 'sharp';
import { Controller, UseGuards, Post, UseInterceptors, UseFilters, UploadedFile, Body, Req } from '@nestjs/common';
import { v4 as uuidv4 } from 'uuid'; // Correct import statement
import { AMQPQueuingService } from '../services/amqp-queuing.service';
import { UploadToFileServerService } from '../services/upload-to-fileServer.service';

@Controller('images')
export class ImageUploadController {
  constructor(private readonly amqpQueuingService: AMQPQueuingService, private readonly uploadToFileServerService: UploadToFileServerService) { }

  @UseGuards(ExternalAuthGuard)
  @Post('classify')
  @UseInterceptors(
    FileInterceptor('image', {
      //storage: memoryStorage(),
      limits: {
        fileSize: 5 * 1024 * 1024, // 5MB (adjust the value as needed)
      },
    }),
  )
  @UseFilters(ImageClassificationExceptionFilter, ImageUploadExceptionFilter)
  async classifyImage(@UploadedFile() image: any, @Body() body: ClassifyImageDto, @Req() request: Request) {
    const classes = body.classes;

    if (!image) {
      throw new ImageUploadException('Image data is missing');
    }

    const resizedAndCompressedImageBuffer = await sharp(image.buffer)
      .resize({ width: 224, height: 224, fit: 'cover' })
      .webp({ quality: 80 })
      .toBuffer();

    // Store the resized image
    try {
      const uploadData = await this.uploadToFileServerService.upload(resizedAndCompressedImageBuffer, image.originalname); //this should send a task to queue
      const taskId = uuidv4();
      await this.amqpQueuingService.send(classes, uploadData.location, 'CLIP', taskId, request); //this should send a task to queue
      return {
        taskId,
      };
    } catch (error) {
      throw new ImageClassificationException('Classification Failed');
    }
  }
}
