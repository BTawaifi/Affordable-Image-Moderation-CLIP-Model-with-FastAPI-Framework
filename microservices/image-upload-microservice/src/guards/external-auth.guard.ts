// ExternalAuthGuard.ts
import {
  CanActivate,
  ExecutionContext,
  Injectable,
  UnauthorizedException,
} from '@nestjs/common';
import { Observable } from 'rxjs';
import { Inject } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices';

@Injectable()
export class ExternalAuthGuard implements CanActivate {
  constructor(@Inject('AUTH_MICROSERVICE') private client: ClientProxy) { }

  canActivate(
    context: ExecutionContext,
  ): boolean | Promise<boolean> | Observable<boolean> {
    const request = context.switchToHttp().getRequest();
    const token = request.headers.authorization?.split(' ')[1];

    if (!token) {
      throw new UnauthorizedException('Missing token');
    }

    return this.client
      .send({ cmd: 'verify-token' }, { token })
      .toPromise()
      .then((isValid: boolean) => {
        if (!isValid) {
          throw new UnauthorizedException('Invalid token');
        }
        return true;
      });
  }
}
