import { Module } from '@nestjs/common';
import { PassportModule } from '@nestjs/passport';
import { JwtModule } from '@nestjs/jwt';
import { AuthController } from './auth.controller';
import { AuthService } from './auth.service';
import { LocalStrategy } from './local.strategy';
import { JwtStrategy } from './jwt.strategy';
import { HttpModule } from '@nestjs/axios';
import { TypeOrmModule } from '@nestjs/typeorm';
import { UserRepository } from '../entity/user/user.repository'; // Import UserRepository

@Module({
  imports: [
    PassportModule,
    JwtModule.register({
      secret: '***REDACTED***',
      signOptions: { expiresIn: '1h' },
    }),
    HttpModule, // Add HttpModule
    TypeOrmModule.forFeature([UserRepository]), // Add UserRepository to the imports
  ],
  controllers: [AuthController],
  providers: [AuthService, LocalStrategy, JwtStrategy, UserRepository], // Add UserRepository to the providers
  exports: [AuthService],
})
export class AuthModule {}
