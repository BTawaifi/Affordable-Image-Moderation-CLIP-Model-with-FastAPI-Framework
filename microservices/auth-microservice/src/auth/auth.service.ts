import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import * as bcrypt from 'bcrypt';
import { User } from 'src/entity/user/user.entity';
import { EntityManager } from 'typeorm';

@Injectable()
export class AuthService {
  constructor(
    private readonly jwtService: JwtService,
    private readonly entityManager: EntityManager, // Inject the EntityManager
  ) {}
  async validateUser(username: string, password: string): Promise<any> {
    const user = await this.entityManager
      .getRepository(User)
      .findOne({ where: { username } });

    if (!user) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const isValidPassword = await bcrypt.compare(password, user.password);
    if (!isValidPassword) {
      throw new UnauthorizedException('Invalid credentials');
    }

    const { password: _, ...result } = user;
    return result;
  }

  async generateHashedPassword(password) {
    const plainPassword = password;
    const saltRounds = 10; // You can change this value according to your security requirements
    const hashedPassword = await bcrypt.hash(plainPassword, saltRounds);
    console.log('Hashed password:', hashedPassword);
    // generateHashedPassword();
  }

  async login(user: any): Promise<any> {
    const payload = { username: user.username, sub: user.id };
    return {
      access_token: this.jwtService.sign(payload),
    };
  }

  // In auth-microservice, add this method to AuthService
  async verifyToken(token: string): Promise<any> {
    try {
      return this.jwtService.verify(token);
    } catch (e) {
      throw new UnauthorizedException('Invalid token');
    }
  }
}
