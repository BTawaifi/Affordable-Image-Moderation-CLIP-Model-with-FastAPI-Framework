import { Catch, ExceptionFilter, ArgumentsHost } from '@nestjs/common';
import { ImageClassificationException } from './exceptions/image-classification.exception';

@Catch(ImageClassificationException)
export class ImageClassificationExceptionFilter implements ExceptionFilter {
  catch(error: ImageClassificationException, host: ArgumentsHost) {
    const response = host.switchToHttp().getResponse();
    response.status(500).json({
      message: 'Failed to classify image',
      error: error.message,
    });
  }
}
