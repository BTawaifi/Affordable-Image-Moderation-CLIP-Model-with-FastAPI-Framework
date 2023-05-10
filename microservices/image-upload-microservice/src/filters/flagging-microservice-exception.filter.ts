import { Catch, ExceptionFilter, ArgumentsHost } from '@nestjs/common';
import { FlaggingMicroserviceException } from './exceptions/flagging-microservice.exception';

@Catch(FlaggingMicroserviceException)
export class FlaggingMicroserviceExceptionFilter implements ExceptionFilter {
  catch(error: FlaggingMicroserviceException, host: ArgumentsHost) {
    const response = host.switchToHttp().getResponse();
    response.status(500).json({
      message: 'Flagging Microservice Issue',
      error: error.message,
    });
  }
}
