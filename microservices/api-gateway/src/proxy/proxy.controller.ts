import {
  Controller,
  Post,
  Req,
  Res,
  UseFilters,
  UseInterceptors,
} from '@nestjs/common';
import { Request, Response } from 'express';
import { HttpExceptionFilter } from '../http-exception.filter';
import { ProxyService } from './proxy.service';

@Controller()
@UseFilters(HttpExceptionFilter)
export class ProxyController {
  constructor(private readonly proxyService: ProxyService) {}

  @UseInterceptors(HttpExceptionFilter)
  @Post('/proxy')
  async handleRequest(@Req() req: Request, @Res() res: Response) {
    const { url, method, body } = req;
    const proxyResponse = await this.proxyService.proxyRequest(url, method, body);

    res.status(proxyResponse.status).json(proxyResponse.data);
  }
}
