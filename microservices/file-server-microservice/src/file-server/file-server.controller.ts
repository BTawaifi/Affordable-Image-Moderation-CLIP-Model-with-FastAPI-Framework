import {
  Controller,
  Get,
  Param,
  Post,
  Res,
  UploadedFile,
  UseFilters,
  UseInterceptors,
} from '@nestjs/common';
import { multerConfig } from './multer.config';
import { FileInterceptor } from '@nestjs/platform-express';
import { ImageSavingExceptionFilter } from 'src/filters/image-saving-exception.filter';
import { ImageSavingException } from 'src/filters/exceptions/image-saving.exception';
import { Response } from 'express';

@Controller('file-server')
export class FileServerController {
  @Post('upload')
  @UseInterceptors(FileInterceptor('image', multerConfig))
  @UseFilters(ImageSavingExceptionFilter)
  async uploadImage(@UploadedFile() image: any) {
    try {
      // Return a response with the uploaded image information
      return {
        message: 'Image uploaded successfully',
        filename: image.filename,
        size: image.size,
        location: `/file-server/images/${image.filename}`,
      };
    } catch (error) {
      // Handle any errors that occur during image upload
      throw new ImageSavingException('Failed to save image');
    }
  }

  @Get('images/:filename')
  async getImage(@Param('filename') filename: string, @Res() res: Response) {
    try {
      const imagePath = `uploads/${filename}`;
      return res.sendFile(imagePath, { root: '.' });
    } catch (error) {
      // Handle any errors that occur during image retrieval
      throw new Error('Failed to get image');
    }
  }
}
