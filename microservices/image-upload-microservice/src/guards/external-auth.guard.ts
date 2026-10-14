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
import { HttpService } from '@nestjs/axios';
import { HttpModule } from '@nestjs/axios';

@Injectable()
export class ExternalAuthGuard implements CanActivate {
  constructor(private httpService: HttpService) {}

  async canActivate(context: ExecutionContext): Promise<boolean> {
    const request = context.switchToHttp().getRequest();
    const token = request.headers.authorization?.split(' ')[1];

    if (!token) {
      throw new UnauthorizedException('Missing token');
    }

    try {
      const response = await this.httpService
        .post('http://localhost:8000/auth/verify-token', { token })
        .toPromise();

      const isValid = response.data.isValid;

      if (!isValid) {
        throw new UnauthorizedException('Invalid token');
      }

      return true;
    } catch (error) {
      throw new UnauthorizedException('Error verifying token');
    }
  }
}
