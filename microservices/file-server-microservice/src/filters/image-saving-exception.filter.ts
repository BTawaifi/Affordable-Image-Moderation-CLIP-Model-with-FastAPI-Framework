import { Catch, ExceptionFilter, ArgumentsHost } from '@nestjs/common';
import { ImageSavingException } from './exceptions/image-saving.exception';

@Catch(ImageSavingException)
export class ImageSavingExceptionFilter implements ExceptionFilter {
  catch(error: ImageSavingException, host: ArgumentsHost) {
    const response = host.switchToHttp().getResponse();
    response.status(400).json({
      message: 'Failed to save image',
      error: error.message,
    });
  }
}
