import { Catch, ExceptionFilter, ArgumentsHost } from '@nestjs/common';
import { ImageUploadException } from './exceptions/image-upload.exception';

@Catch(ImageUploadException)
export class ImageUploadExceptionFilter implements ExceptionFilter {
  catch(error: ImageUploadException, host: ArgumentsHost) {
    const response = host.switchToHttp().getResponse();
    response.status(400).json({
      message: 'Failed to upload image',
      error: error.message,
    });
  }
}
