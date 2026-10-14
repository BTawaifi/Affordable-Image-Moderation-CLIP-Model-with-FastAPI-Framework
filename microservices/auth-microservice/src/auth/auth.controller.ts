// AuthController.ts
import {
  Body,
  Controller,
  Post,
  Request,
  UnauthorizedException,
  UseGuards,
} from '@nestjs/common';
import { AuthGuard } from '@nestjs/passport';
import { AuthService } from './auth.service';
import { JwtService } from '@nestjs/jwt';

@Controller('auth')
export class AuthController {
  constructor(
    private readonly authService: AuthService,
    private readonly jwtService: JwtService,
  ) {}

  @UseGuards(AuthGuard('local'))
  @Post('login')
  async login(@Request() req): Promise<any> {
    return this.authService.login(req.user);
  }

  // In auth-microservice, add this method to AuthController
  @Post('verify-token')
  async verifyToken(@Body() body: { token: string }): Promise<any> {
    try {
      const { token } = body;
      const decodedToken = this.jwtService.verify(token);
      return {
        userId: decodedToken.sub,
        username: decodedToken.username,
        isValid: true,
      };
    } catch (e) {
      throw new UnauthorizedException('Invalid token');
    }
  }
}
